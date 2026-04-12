from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import copy
import json
import os
import statistics

import optuna
import torch

from torchkit.data._dataset import TorchkitDataset
from torchkit.train._event_log import JsonlEventLogger
from torchkit.train.cv._base_cv import (
    _aggregate_report_results,
    _clone_state_dict_cpu,
    _clone_tensor_dict,
    _concat_tensor_dicts,
    _resolve_original_indices_for_subset,
)
from torchkit.train.cv._optuna_results import FoldResult, OptunaTrialResult
from torchkit.train.cv.optuna_search_cv import OptunaSearchCV
from torchkit.train.trainer import EpochResult, Trainer


@dataclass
class _LiveFoldRunner:
    fold: int
    trainer: Optional[Trainer]
    val_subset: Any
    train_indices: list[int]
    val_indices: list[int]
    iterator: Iterator[EpochResult]
    log_file: Optional[str] = None
    completed: bool = False
    last_event: Optional[EpochResult] = None
    last_selection_score: Optional[float] = None
    fold_result: Optional[FoldResult] = None


class _PrunedTrialWithResult(optuna.TrialPruned):
    def __init__(self, trial_result: OptunaTrialResult, message: str):
        super().__init__(message)
        self.trial_result = trial_result


class InMemoryOptunaSearchCV(OptunaSearchCV):
    """
    Optuna CV engine that keeps one trainer per fold alive in memory and advances
    all folds epoch-by-epoch via ``Trainer.fit_iter()``.
    """

    def _build_live_folds(
        self,
        *,
        trial: optuna.Trial,
        params: dict[str, Any],
        search_dataset: TorchkitDataset,
        search_index: Any,
        search_groups: Any,
        trial_logger: Optional[JsonlEventLogger],
    ) -> list[_LiveFoldRunner]:
        live_folds: list[_LiveFoldRunner] = []

        for fold, (train_subset, val_subset) in enumerate(
            self._split(self.splitter, search_dataset, search_index, search_groups)
        ):
            train_original_indices = _resolve_original_indices_for_subset(train_subset)
            val_original_indices = _resolve_original_indices_for_subset(val_subset)

            train_loader = self.dataloader_factory(train_subset, True)
            val_loader = self.dataloader_factory(val_subset, False)

            _, _, trainer = self._build_trainer_for_trial(params=params)

            fold_log_file = None
            if trial_logger is not None and self.log_dir is not None:
                trial_logger.emit(
                    "cv_fold_start",
                    payload={
                        "fold": fold,
                        "n_train": len(train_original_indices),
                        "n_val": len(val_original_indices),
                    },
                    message=(
                        f"Trial {trial.number} fold {fold} started "
                        f"(n_train={len(train_original_indices)}, n_val={len(val_original_indices)})."
                    ),
                )
                fold_log_file = os.path.join(
                    self.log_dir,
                    "trials",
                    f"trial_{trial.number:03d}_fold_{fold:03d}_trainer.log.jsonl",
                )
                trainer._set_event_logger(
                    JsonlEventLogger(
                        fold_log_file,
                        scope="trainer",
                        echo_console=True,
                        context={"trial_number": trial.number, "fold": fold},
                    )
                )

            live_folds.append(
                _LiveFoldRunner(
                    fold=fold,
                    trainer=trainer,
                    val_subset=val_subset,
                    train_indices=copy.deepcopy(train_original_indices),
                    val_indices=copy.deepcopy(val_original_indices),
                    iterator=trainer.fit_iter(
                        train_loader,
                        val_loader,
                        trial=None,
                        reset_state=True,
                    ),
                    log_file=fold_log_file,
                )
            )

        return live_folds

    def _release_live_folds(self, live_folds: list[_LiveFoldRunner]) -> None:
        for fold_runner in live_folds:
            self._release_trainer_resources(fold_runner.trainer)
            fold_runner.trainer = None
            fold_runner.iterator = iter(())
            fold_runner.last_event = None
        live_folds.clear()
        self._cleanup_cuda_cache()

    def _collect_fold_result(self, *, fold_runner: _LiveFoldRunner) -> FoldResult:
        trainer = fold_runner.trainer
        if trainer is None:
            raise RuntimeError(f"Fold {fold_runner.fold} trainer has already been released.")
        fold_report_results = self._evaluate_report(trainer, fold_runner.val_subset)
        metric = trainer.state.best_metric
        if metric is not None:
            metric = float(metric)

        return FoldResult(
            fold=fold_runner.fold,
            train_indices=copy.deepcopy(fold_runner.train_indices),
            val_indices=copy.deepcopy(fold_runner.val_indices),
            best_metric=metric,
            best_epoch=trainer.state.best_epoch,
            best_state_dict_cpu=_clone_state_dict_cpu(trainer.state.best_state_dict_cpu),
            oof_logits=_clone_tensor_dict(trainer.state.oof_logits),
            oof_targets=_clone_tensor_dict(trainer.state.oof_targets),
            oof_sample_indices=copy.deepcopy(fold_runner.val_indices),
            report_results=copy.deepcopy(fold_report_results),
            log_file=fold_runner.log_file,
        )

    def _materialize_fold_result(self, *, fold_runner: _LiveFoldRunner) -> FoldResult:
        if fold_runner.fold_result is None:
            fold_runner.fold_result = self._collect_fold_result(fold_runner=fold_runner)
        return fold_runner.fold_result

    def _release_fold_runner(self, fold_runner: _LiveFoldRunner) -> None:
        self._release_trainer_resources(fold_runner.trainer)
        fold_runner.trainer = None
        fold_runner.iterator = iter(())
        fold_runner.last_event = None

    def _collect_trial_result(
        self,
        *,
        trial: optuna.Trial,
        params: dict[str, Any],
        live_folds: list[_LiveFoldRunner],
        intermediate_reports: list[dict[str, Any]],
        status: str,
        trial_log_file: Optional[str],
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
        pruned_epoch: Optional[int] = None,
        trial_logger: Optional[JsonlEventLogger] = None,
    ) -> OptunaTrialResult:
        fold_results = [self._materialize_fold_result(fold_runner=fold_runner) for fold_runner in live_folds]

        fold_metrics = [
            float(fold_result.best_metric)
            for fold_result in fold_results
            if fold_result.best_metric is not None
        ]
        fold_selection_scores = [self._to_selection_score(metric) for metric in fold_metrics]

        aggregate_metric = (
            float(sum(fold_metrics) / len(fold_metrics))
            if fold_metrics
            else None
        )
        aggregate_selection_score = (
            float(sum(fold_selection_scores) / len(fold_selection_scores))
            if fold_selection_scores
            else None
        )

        fold_report_results_all = [copy.deepcopy(fold_result.report_results) for fold_result in fold_results]
        fold_oof_logits_all = [fold_result.oof_logits for fold_result in fold_results if fold_result.oof_logits]
        fold_oof_targets_all = [fold_result.oof_targets for fold_result in fold_results if fold_result.oof_targets]
        aggregate_oof_sample_indices: list[int] = []
        for fold_result in fold_results:
            if fold_result.oof_logits or fold_result.oof_targets:
                aggregate_oof_sample_indices.extend(fold_result.oof_sample_indices)

        trial_result = OptunaTrialResult(
            trial_number=trial.number,
            params=copy.deepcopy(params),
            status=status,
            aggregate_metric=aggregate_metric,
            aggregate_selection_score=aggregate_selection_score,
            intermediate_reports=copy.deepcopy(intermediate_reports),
            pruned_epoch=pruned_epoch,
            fold_results=fold_results,
            aggregate_fold_report_results=_aggregate_report_results(fold_report_results_all),
            log_file=trial_log_file,
            aggregate_oof_logits=_concat_tensor_dicts(fold_oof_logits_all) if fold_oof_logits_all else {},
            aggregate_oof_targets=_concat_tensor_dicts(fold_oof_targets_all) if fold_oof_targets_all else {},
            aggregate_oof_sample_indices=copy.deepcopy(aggregate_oof_sample_indices),
            error_message=error_message,
            error_traceback=error_traceback,
        )

        if status == "SUCCESS":
            search_original_indices = sorted(
                idx for fold_result in fold_results for idx in fold_result.val_indices
            )
            self._assert_exact_oof_coverage(
                sample_indices=aggregate_oof_sample_indices,
                reference_indices=search_original_indices,
                context=f"Trial {trial.number}",
            )

        if trial_logger is not None:
            for fold_result in fold_results:
                trial_logger.emit(
                    "cv_fold_end",
                    payload={
                        "fold": fold_result.fold,
                        "best_epoch": fold_result.best_epoch,
                        "best_metric": fold_result.best_metric,
                        "selection_score": None if fold_result.best_metric is None else self._to_selection_score(float(fold_result.best_metric)),
                        "log_file": fold_result.log_file,
                    },
                    message=(
                        f"Trial {trial.number} fold {fold_result.fold} ended. "
                        f"best_epoch={fold_result.best_epoch}, best_metric={fold_result.best_metric}, "
                        f"trainer_log={fold_result.log_file}."
                    ),
                )
            trial_logger.emit(
                "cv_trial_end",
                payload={
                    "trial_number": trial.number,
                    "status": status,
                    "aggregate_metric": aggregate_metric,
                    "aggregate_selection_score": aggregate_selection_score,
                    "pruned_epoch": pruned_epoch,
                },
                message=(
                    f"Trial {trial.number} ended with status={status}. "
                    f"aggregate_metric={aggregate_metric}, "
                    f"aggregate_selection_score={aggregate_selection_score}."
                ),
            )

        return trial_result

    def _run_single_trial_with_params(
        self,
        *,
        trial_number: int,
        params: dict[str, Any],
        search_dataset: TorchkitDataset,
        search_index: Any,
        search_groups: Any,
        search_original_indices: list[int],
        trial: optuna.Trial | None = None,
    ) -> OptunaTrialResult:
        del search_original_indices

        trial_logger = None
        trial_log_file = None
        strategy = self._distributed_strategy()
        if self.logging and self.log_dir is not None and self._is_main_process():
            trial_log_file = os.path.join(self.log_dir, "trials", f"trial_{trial_number:03d}.log.jsonl")
            trial_logger = JsonlEventLogger(
                trial_log_file,
                scope="optuna_trial",
                echo_console=True,
                context={"trial_number": trial_number},
            )
            trial_logger.emit(
                "cv_trial_start",
                payload={"trial_number": trial_number, "params": copy.deepcopy(params)},
                message=(
                    f"Trial {trial_number} started with params "
                    f"{json.dumps(params, sort_keys=True)}. "
                    f"Logging to {trial_log_file}."
                ),
            )

        live_folds = self._build_live_folds(
            trial=type("_TrialView", (), {"number": trial_number})(),
            params=params,
            search_dataset=search_dataset,
            search_index=search_index,
            search_groups=search_groups,
            trial_logger=trial_logger,
        )
        intermediate_reports: list[dict[str, Any]] = []

        try:
            while True:
                progressed = False
                validation_epochs: set[int] = set()

                for fold_runner in live_folds:
                    if fold_runner.completed:
                        continue

                    try:
                        event = next(fold_runner.iterator)
                    except StopIteration:
                        fold_runner.completed = True
                        fold_runner.fold_result = self._collect_fold_result(fold_runner=fold_runner)
                        self._release_fold_runner(fold_runner)
                        continue

                    progressed = True
                    fold_runner.last_event = event
                    if event.did_validate and event.selection_score is not None:
                        fold_runner.last_selection_score = float(event.selection_score)
                        validation_epochs.add(int(event.epoch))

                if strategy is not None:
                    # Keep all ranks at the same trial-sync boundary before any rank
                    # starts the next epoch step, prunes, or exits the trial loop.
                    strategy.barrier()

                if not progressed:
                    break

                if not validation_epochs:
                    continue

                if len(validation_epochs) != 1:
                    raise RuntimeError(
                        "In-memory synchronized CV expected a single validation epoch per sync step, "
                        f"got {sorted(validation_epochs)}."
                    )

                sync_epoch = next(iter(validation_epochs))
                fold_selection_scores = [
                    float(fold_runner.last_selection_score)
                    for fold_runner in live_folds
                    if fold_runner.last_selection_score is not None
                ]
                if len(fold_selection_scores) == 0:
                    continue

                aggregate_selection_score = float(sum(fold_selection_scores) / len(fold_selection_scores))
                report = {
                    "epoch": sync_epoch,
                    "aggregate_selection_score": aggregate_selection_score,
                    "fold_selection_scores": copy.deepcopy(fold_selection_scores),
                    "n_reporting_folds": len(fold_selection_scores),
                    "n_completed_folds": sum(1 for fold_runner in live_folds if fold_runner.completed),
                }
                intermediate_reports.append(report)

                if trial_logger is not None:
                    trial_logger.emit(
                        "cv_trial_epoch_report",
                        payload=copy.deepcopy(report),
                        message=(
                            f"Trial {trial_number} sync epoch {sync_epoch}: "
                            f"aggregate_selection_score={aggregate_selection_score:.6f}."
                        ),
                    )

                should_prune = False
                if trial is not None and self._is_main_process():
                    trial.report(aggregate_selection_score, sync_epoch)
                    should_prune = bool(trial.should_prune())
                if strategy is not None:
                    should_prune = bool(strategy.broadcast_object(should_prune, src=0))
                if should_prune:
                    if strategy is not None:
                        strategy.barrier()
                    raise _PrunedTrialWithResult(
                        self._collect_trial_result(
                            trial=type("_TrialView", (), {"number": trial_number})(),
                            params=params,
                            live_folds=live_folds,
                            intermediate_reports=intermediate_reports,
                            status="PRUNED",
                            trial_log_file=trial_log_file,
                            error_message=f"Trial pruned at epoch {sync_epoch}.",
                            pruned_epoch=sync_epoch,
                            trial_logger=trial_logger,
                        ),
                        message=f"Trial pruned at epoch {sync_epoch}.",
                    )

            if strategy is not None:
                strategy.barrier()
            return self._collect_trial_result(
                trial=type("_TrialView", (), {"number": trial_number})(),
                params=params,
                live_folds=live_folds,
                intermediate_reports=intermediate_reports,
                status="SUCCESS",
                trial_log_file=trial_log_file,
                trial_logger=trial_logger,
            )
        finally:
            self._release_live_folds(live_folds)

    def _run_single_trial(
        self,
        *,
        trial: optuna.Trial,
        search_dataset: TorchkitDataset,
        search_index: Any,
        search_groups: Any,
        search_original_indices: list[int],
    ) -> OptunaTrialResult:
        params = self.suggest_parameters(trial, self.parameter_grid)
        return self._run_single_trial_with_params(
            trial_number=trial.number,
            params=params,
            search_dataset=search_dataset,
            search_index=search_index,
            search_groups=search_groups,
            search_original_indices=search_original_indices,
            trial=trial,
        )

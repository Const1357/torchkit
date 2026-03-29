from __future__ import annotations
import os

from typing import Any, Optional

import copy
import statistics
import traceback

import optuna
from optuna.trial import TrialState
import torch

from torchkit.data._dataset import TorchkitDataset
from torchkit.evaluate.report.bundle import BundleReportEvaluator
from torchkit.train._event_log import JsonlEventLogger
from torchkit.train.cv._base_cv import (
    _aggregate_report_results,
    _clone_state_dict_cpu,
    _clone_tensor_dict,
    _concat_tensor_dicts,
    _resolve_original_indices_for_subset,
    _safe_take,
)
from torchkit.train.cv._base_search_cv import BaseSearchCV
from torchkit.train.cv._optuna_results import (
    FoldResult,
    OptunaSearchCVResult,
    OptunaTrialResult,
)
from torchkit.train.cv._optuna_search_mixin import (
    OptunaSearchMixin,
    ParameterGrid,
)
from torchkit.train.trainer import Trainer
from torchkit.distributed import DDPStrategy


class OptunaSearchCV(OptunaSearchMixin, BaseSearchCV):
    """
    Reusable single-study Optuna CV engine.

    This is the core search primitive:
    - one study
    - one CV splitter
    - one training/search pool
    - optional final refit on the full search pool
    - optional holdout evaluation

    NestedOptunaSearchCV composes this engine inside its outer loop.
    """

    def __init__(
        self,
        *,
        model_spec,
        trainer_spec,
        parameter_grid: ParameterGrid,
        splitter_cls,
        dataloader_factory=None,
        n_trials: int = 10,
        max_trial_attempts: Optional[int] = None,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        calibrate: bool = True,
        report_evaluator: Optional[BundleReportEvaluator] = None,
        logging: bool = False,
        _log_root_dir: Optional[str] = None,
        final_model_dir: Optional[str] = None,
        keep_final_model_state_dict_cpu: bool = True,
    ):
        super().__init__(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            parameter_grid=parameter_grid,
            splitter_cls=splitter_cls,
            dataloader_factory=dataloader_factory,
            n_trials=n_trials,
            max_trial_attempts=max_trial_attempts,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            calibrate=calibrate,
            report_evaluator=report_evaluator,
            logging=logging,
            _log_root_dir=_log_root_dir,
            final_model_dir=final_model_dir,
            keep_final_model_state_dict_cpu=keep_final_model_state_dict_cpu,
        )

    def _distributed_strategy(self) -> Optional[DDPStrategy]:
        strategy = getattr(self.trainer_spec, "distributed_strategy", None)
        if strategy is None or not strategy.is_enabled:
            return None
        return strategy

    def _is_main_process(self) -> bool:
        strategy = self._distributed_strategy()
        if strategy is None:
            return True
        return strategy.is_main_process

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
        del trial
        _, _, trainer = self._build_trainer_for_trial(params=params)
        trial_logger = None
        trial_log_file = None
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
                message=f"Trial {trial_number} started. Logging to {trial_log_file}.",
            )

        fold_results: list[FoldResult] = []
        fold_metrics: list[float] = []
        fold_selection_scores: list[float] = []
        fold_report_results_all: list[Optional[dict[str, Any]]] = []

        fold_oof_logits_all: list[dict[str, torch.Tensor]] = []
        fold_oof_targets_all: list[dict[str, torch.Tensor]] = []
        aggregate_oof_sample_indices: list[int] = []

        for fold, (train_subset, val_subset) in enumerate(
            self._split(self.splitter, search_dataset, search_index, search_groups)
        ):
            train_original_indices = _resolve_original_indices_for_subset(train_subset)
            val_original_indices = _resolve_original_indices_for_subset(val_subset)

            train_loader = self.dataloader_factory(train_subset, True)
            val_loader = self.dataloader_factory(val_subset, False)
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
                        f"Trial {trial_number} fold {fold} started "
                        f"(n_train={len(train_original_indices)}, n_val={len(val_original_indices)})."
                    ),
                )
                fold_log_file = os.path.join(
                    self.log_dir,
                    "trials",
                    f"trial_{trial_number:03d}_fold_{fold:03d}_trainer.log.jsonl",
                )
                trainer._set_event_logger(
                    JsonlEventLogger(
                        fold_log_file,
                        scope="trainer",
                        echo_console=True,
                        context={"trial_number": trial_number, "fold": fold},
                    )
                )

            trainer.reset_state()
            trainer.fit(
                train_loader,
                val_loader,
                trial=None,
            )

            fold_report_results = self._evaluate_report(trainer, val_subset)

            metric = trainer.state.best_metric
            if metric is not None:
                metric = float(metric)

            fold_result = FoldResult(
                fold=fold,
                train_indices=copy.deepcopy(train_original_indices),
                val_indices=copy.deepcopy(val_original_indices),
                best_metric=metric,
                best_epoch=trainer.state.best_epoch,
                best_state_dict_cpu=_clone_state_dict_cpu(trainer.state.best_state_dict_cpu),
                oof_logits=_clone_tensor_dict(trainer.state.oof_logits),
                oof_targets=_clone_tensor_dict(trainer.state.oof_targets),
                oof_sample_indices=copy.deepcopy(val_original_indices),
                report_results=copy.deepcopy(fold_report_results),
                log_file=fold_log_file,
            )
            fold_results.append(fold_result)
            fold_report_results_all.append(copy.deepcopy(fold_report_results))
            if trial_logger is not None:
                trial_logger.emit(
                    "cv_fold_end",
                    payload={
                        "fold": fold,
                        "best_epoch": trainer.state.best_epoch,
                        "best_metric": trainer.state.best_metric,
                        "selection_score": None if metric is None else self._to_selection_score(metric),
                        "log_file": fold_log_file,
                    },
                    message=(
                        f"Trial {trial_number} fold {fold} ended. "
                        f"best_epoch={trainer.state.best_epoch}, best_metric={metric}, "
                        f"trainer_log={fold_log_file}."
                    ),
                )

            if metric is not None:
                fold_metrics.append(metric)
                fold_selection_scores.append(self._to_selection_score(metric))

            if trainer.state.oof_logits:
                fold_oof_logits_all.append(_clone_tensor_dict(trainer.state.oof_logits))
            if trainer.state.oof_targets:
                fold_oof_targets_all.append(_clone_tensor_dict(trainer.state.oof_targets))
            if trainer.state.oof_logits or trainer.state.oof_targets:
                aggregate_oof_sample_indices.extend(val_original_indices)

        if len(fold_metrics) == 0:
            raise ValueError(f"Trial {trial_number} produced no valid fold metrics.")

        aggregate_metric = sum(fold_metrics) / len(fold_metrics)
        aggregate_selection_score = sum(fold_selection_scores) / len(fold_selection_scores)

        aggregate_oof_logits = _concat_tensor_dicts(fold_oof_logits_all) if fold_oof_logits_all else {}
        aggregate_oof_targets = _concat_tensor_dicts(fold_oof_targets_all) if fold_oof_targets_all else {}

        self._assert_exact_oof_coverage(
            sample_indices=aggregate_oof_sample_indices,
            reference_indices=search_original_indices,
            context=f"Trial {trial_number}",
        )

        trial_result = OptunaTrialResult(
            trial_number=trial_number,
            params=copy.deepcopy(params),
            status="SUCCESS",
            aggregate_metric=float(aggregate_metric),
            aggregate_selection_score=float(aggregate_selection_score),
            fold_results=fold_results,
            aggregate_fold_report_results=_aggregate_report_results(fold_report_results_all),
            log_file=trial_log_file,
            aggregate_oof_logits=aggregate_oof_logits,
            aggregate_oof_targets=aggregate_oof_targets,
            aggregate_oof_sample_indices=copy.deepcopy(aggregate_oof_sample_indices),
            error_message=None,
            error_traceback=None,
        )
        if trial_logger is not None:
            trial_logger.emit(
                "cv_trial_end",
                payload={
                    "trial_number": trial_number,
                    "status": "SUCCESS",
                    "aggregate_metric": aggregate_metric,
                    "aggregate_selection_score": aggregate_selection_score,
                },
                message=(
                    f"Trial {trial_number} ended successfully. "
                    f"aggregate_metric={aggregate_metric:.6f}, "
                    f"aggregate_selection_score={aggregate_selection_score:.6f}."
                ),
            )
        return trial_result

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
        )

    def _default_trial_log_file(self, *, trial_number: int) -> Optional[str]:
        if not self.logging or self.log_dir is None:
            return None
        return os.path.join(self.log_dir, "trials", f"trial_{trial_number:03d}.log.jsonl")

    def _build_pruned_trial_result(
        self,
        *,
        trial_number: int,
        params: dict[str, Any],
        exception: BaseException,
        traceback_text: str,
    ) -> OptunaTrialResult:
        provided = getattr(exception, "trial_result", None)
        if isinstance(provided, OptunaTrialResult):
            if provided.error_message is None:
                provided.error_message = str(exception)
            if provided.error_traceback is None:
                provided.error_traceback = traceback_text
            if provided.log_file is None:
                provided.log_file = self._default_trial_log_file(trial_number=trial_number)
            return provided

        return OptunaTrialResult(
            trial_number=trial_number,
            params=copy.deepcopy(params),
            status="PRUNED",
            aggregate_metric=None,
            aggregate_selection_score=None,
            intermediate_reports=[],
            pruned_epoch=None,
            fold_results=[],
            aggregate_fold_report_results=None,
            log_file=self._default_trial_log_file(trial_number=trial_number),
            aggregate_oof_logits={},
            aggregate_oof_targets={},
            aggregate_oof_sample_indices=[],
            error_message=str(exception),
            error_traceback=traceback_text,
        )

    def _build_failed_trial_result(
        self,
        *,
        trial_number: int,
        params: dict[str, Any],
        exception: BaseException,
        traceback_text: str,
    ) -> OptunaTrialResult:
        return OptunaTrialResult(
            trial_number=trial_number,
            params=copy.deepcopy(params),
            status="FAILED",
            aggregate_metric=None,
            aggregate_selection_score=None,
            intermediate_reports=[],
            pruned_epoch=None,
            fold_results=[],
            aggregate_fold_report_results=None,
            log_file=self._default_trial_log_file(trial_number=trial_number),
            aggregate_oof_logits={},
            aggregate_oof_targets={},
            aggregate_oof_sample_indices=[],
            error_message=f"{type(exception).__name__}: {exception}",
            error_traceback=traceback_text,
        )

    def run(
        self,
        dataset: TorchkitDataset,
        index: Any = None,
        groups: Optional[Any] = None,
        *,
        holdout_dataset: Optional[TorchkitDataset] = None,
    ) -> OptunaSearchCVResult:
        search_original_indices = _resolve_original_indices_for_subset(dataset)

        search_index = _safe_take(index, search_original_indices) if index is not None else None
        search_groups = _safe_take(groups, search_original_indices) if groups is not None else None

        selection_metric_name = self._selection_metric_name()
        selection_metric_direction = self._selection_metric_direction()
        selection_metric_spec = self._selection_metric_spec()
        strategy = self._distributed_strategy()
        if strategy is not None:
            strategy.initialize()
        is_main_process = self._is_main_process()

        trial_results: list[OptunaTrialResult] = []
        run_logger = None
        run_log_file = None
        if self.logging and self.log_dir is not None and is_main_process:
            run_log_file = os.path.join(self.log_dir, "search.log.jsonl")
            run_logger = JsonlEventLogger(
                run_log_file,
                scope="optuna_search_cv",
                echo_console=True,
            )
            run_logger.emit(
                "cv_run_start",
                payload={
                    "n_trials": self.n_trials,
                    "n_splits": self.n_splits,
                    "splitter_name": self.splitter_cls.__name__,
                    "search_pool_size": len(search_original_indices),
                    "has_holdout": holdout_dataset is not None,
                    "log_dir": self.log_dir,
                },
                message=(
                    f"OptunaSearchCV started: n_trials={self.n_trials}, n_splits={self.n_splits}, "
                    f"splitter={self.splitter_cls.__name__}. Logging to {run_log_file}."
                ),
            )

        study = self._create_study() if is_main_process else None

        attempted_trials = 0
        successful_trials = 0
        failed_trials = 0
        pruned_trials = 0

        def _exhausted_attempts_error() -> RuntimeError:
            last_error = None
            if trial_results:
                last_error = trial_results[-1].error_message
            suffix = "" if last_error is None else f" Last trial error: {last_error}"
            return RuntimeError(
                f"Reached max_trial_attempts={self.max_trial_attempts} before obtaining "
                f"{self.n_trials} successful trials. "
                f"Successful={successful_trials}, failed={failed_trials}, pruned={pruned_trials}."
                f"{suffix}"
            )

        if strategy is None:
            while successful_trials < self.n_trials:
                if attempted_trials >= self.max_trial_attempts:
                    raise _exhausted_attempts_error()

                assert study is not None
                trial = study.ask()
                attempted_trials += 1
                params = copy.deepcopy(self.suggest_parameters(trial, self.parameter_grid))

                try:
                    trial_result = self._run_single_trial(
                        trial=trial,
                        search_dataset=dataset,
                        search_index=search_index,
                        search_groups=search_groups,
                        search_original_indices=search_original_indices,
                    )
                    assert trial_result.aggregate_selection_score is not None
                    study.tell(trial, trial_result.aggregate_selection_score)
                    trial_results.append(trial_result)
                    successful_trials += 1
                    if run_logger is not None:
                        run_logger.emit(
                            "cv_trial_recorded",
                            payload={
                                "trial_number": trial_result.trial_number,
                                "status": trial_result.status,
                                "aggregate_metric": trial_result.aggregate_metric,
                                "aggregate_selection_score": trial_result.aggregate_selection_score,
                                "trial_log_file": trial_result.log_file,
                            },
                            message=(
                                f"Recorded successful trial {trial_result.trial_number} "
                                f"(aggregate_metric={trial_result.aggregate_metric}, "
                                f"trial_log={trial_result.log_file})."
                            ),
                        )

                except optuna.TrialPruned as e:
                    tb = traceback.format_exc()
                    study.tell(trial, state=TrialState.PRUNED)
                    pruned_result = self._build_pruned_trial_result(
                        trial_number=trial.number,
                        params=params,
                        exception=e,
                        traceback_text=tb,
                    )
                    trial_results.append(pruned_result)
                    pruned_trials += 1
                    if run_logger is not None:
                        run_logger.emit(
                            "cv_trial_recorded",
                            payload={
                                "trial_number": trial.number,
                                "status": "PRUNED",
                                "error_message": str(e),
                                "trial_log_file": pruned_result.log_file,
                            },
                            message=f"Trial {trial.number} pruned: {e}",
                        )

                except Exception as e:
                    tb = traceback.format_exc()
                    study.tell(trial, state=TrialState.FAIL)
                    failed_result = self._build_failed_trial_result(
                        trial_number=trial.number,
                        params=params,
                        exception=e,
                        traceback_text=tb,
                    )
                    trial_results.append(failed_result)
                    failed_trials += 1
                    if run_logger is not None:
                        run_logger.emit(
                            "cv_trial_recorded",
                            payload={
                                "trial_number": trial.number,
                                "status": "FAILED",
                                "error_message": f"{type(e).__name__}: {e}",
                                "trial_log_file": failed_result.log_file,
                            },
                            message=f"Trial {trial.number} failed with {type(e).__name__}: {e}",
                        )
        else:
            while True:
                command: dict[str, Any] | None = None
                trial = None
                if is_main_process:
                    assert study is not None
                    if successful_trials >= self.n_trials:
                        command = {"type": "stop"}
                    else:
                        if attempted_trials >= self.max_trial_attempts:
                            raise _exhausted_attempts_error()
                        trial = study.ask()
                        command = {
                            "type": "trial",
                            "trial_number": trial.number,
                            "params": self.suggest_parameters(trial, self.parameter_grid),
                        }

                command = strategy.broadcast_object(command, src=0)

                assert command is not None
                if command["type"] == "stop":
                    break

                trial_number = int(command["trial_number"])
                params = copy.deepcopy(command["params"])
                attempted_trials += 1

                try:
                    trial_result = self._run_single_trial_with_params(
                        trial_number=trial_number,
                        params=params,
                        search_dataset=dataset,
                        search_index=search_index,
                        search_groups=search_groups,
                        search_original_indices=search_original_indices,
                        trial=trial,
                    )
                    assert trial_result.aggregate_selection_score is not None
                    if is_main_process:
                        assert study is not None
                        assert trial is not None
                        study.tell(trial, trial_result.aggregate_selection_score)
                    trial_results.append(trial_result)
                    successful_trials += 1
                    if run_logger is not None:
                        run_logger.emit(
                            "cv_trial_recorded",
                            payload={
                                "trial_number": trial_result.trial_number,
                                "status": trial_result.status,
                                "aggregate_metric": trial_result.aggregate_metric,
                                "aggregate_selection_score": trial_result.aggregate_selection_score,
                                "trial_log_file": trial_result.log_file,
                            },
                            message=(
                                f"Recorded successful trial {trial_result.trial_number} "
                                f"(aggregate_metric={trial_result.aggregate_metric}, "
                                f"trial_log={trial_result.log_file})."
                            ),
                        )

                except optuna.TrialPruned as e:
                    tb = traceback.format_exc()
                    if is_main_process:
                        assert study is not None
                        assert trial is not None
                        study.tell(trial, state=TrialState.PRUNED)
                    pruned_result = self._build_pruned_trial_result(
                        trial_number=trial_number,
                        params=params,
                        exception=e,
                        traceback_text=tb,
                    )
                    trial_results.append(pruned_result)
                    pruned_trials += 1
                    if run_logger is not None:
                        run_logger.emit(
                            "cv_trial_recorded",
                            payload={
                                "trial_number": trial_number,
                                "status": "PRUNED",
                                "error_message": str(e),
                                "trial_log_file": pruned_result.log_file,
                            },
                            message=f"Trial {trial_number} pruned: {e}",
                        )

                except Exception as e:
                    tb = traceback.format_exc()
                    if is_main_process:
                        assert study is not None
                        assert trial is not None
                        study.tell(trial, state=TrialState.FAIL)
                    failed_result = self._build_failed_trial_result(
                        trial_number=trial_number,
                        params=params,
                        exception=e,
                        traceback_text=tb,
                    )
                    trial_results.append(failed_result)
                    failed_trials += 1
                    if run_logger is not None:
                        run_logger.emit(
                            "cv_trial_recorded",
                            payload={
                                "trial_number": trial_number,
                                "status": "FAILED",
                                "error_message": f"{type(e).__name__}: {e}",
                                "trial_log_file": failed_result.log_file,
                            },
                            message=f"Trial {trial_number} failed with {type(e).__name__}: {e}",
                        )

        successful_trial_results = [tr for tr in trial_results if tr.status == "SUCCESS"]
        if len(successful_trial_results) == 0:
            raise RuntimeError("OptunaSearchCV produced no successful trials.")

        best_trial_number = None
        if is_main_process:
            assert study is not None
            best_trial_number = study.best_trial.number
        if strategy is not None:
            best_trial_number = strategy.broadcast_object(best_trial_number, src=0)
        assert best_trial_number is not None

        try:
            best_trial_result = next(
                tr for tr in successful_trial_results if tr.trial_number == best_trial_number
            )
        except StopIteration as e:
            raise RuntimeError(
                f"Best Optuna trial {best_trial_number} was not found in stored successful trial_results."
            ) from e

        assert best_trial_result.aggregate_metric is not None
        assert best_trial_result.aggregate_selection_score is not None

        best_params = copy.deepcopy(best_trial_result.params)
        best_metric = float(best_trial_result.aggregate_metric)
        best_selection_score = float(best_trial_result.aggregate_selection_score)

        selected_fold_metrics = [
            float(r.best_metric) for r in best_trial_result.fold_results if r.best_metric is not None
        ]
        selected_metric_mean = (
            float(statistics.mean(selected_fold_metrics)) if selected_fold_metrics else None
        )
        selected_metric_std = (
            float(statistics.stdev(selected_fold_metrics))
            if len(selected_fold_metrics) >= 2
            else 0.0 if len(selected_fold_metrics) == 1
            else None
        )
        selected_metric_min = (
            float(min(selected_fold_metrics)) if selected_fold_metrics else None
        )
        selected_metric_max = (
            float(max(selected_fold_metrics)) if selected_fold_metrics else None
        )

        final_model_spec, final_trainer_spec, final_trainer = self._build_trainer_for_trial(params=best_params)
        search_loader = self.dataloader_factory(dataset, True)
        final_refit_log_file = None
        if self.logging and self.log_dir is not None:
            final_refit_log_file = os.path.join(self.log_dir, "final_refit_trainer.log.jsonl")
            final_trainer._set_event_logger(
                JsonlEventLogger(
                    final_refit_log_file,
                    scope="trainer",
                    echo_console=True,
                    context={"stage": "final_refit"},
                )
            )
            if run_logger is not None:
                run_logger.emit(
                    "cv_final_refit_start",
                    payload={
                        "best_trial_number": best_trial_number,
                        "best_params": copy.deepcopy(best_params),
                    },
                    message=f"Final refit started for best trial {best_trial_number}.",
                )

        fold_best_epochs = [r.best_epoch for r in best_trial_result.fold_results if r.best_epoch is not None]
        final_fit_epochs = (
            int(statistics.median(fold_best_epochs))
            if fold_best_epochs
            else int(final_trainer.config.max_epochs)
        )

        final_trainer.fit(
            search_loader,
            val_loader=None,
            reset_state=True,
            max_epochs=final_fit_epochs,
            early_stopping_patience=None,
        )
        if run_logger is not None:
            run_logger.emit(
                "cv_final_refit_end",
                payload={
                    "best_trial_number": best_trial_number,
                    "final_fit_epochs": final_fit_epochs,
                    "final_best_epoch": final_trainer.state.best_epoch,
                    "final_best_metric": final_trainer.state.best_metric,
                    "trainer_log_file": final_refit_log_file,
                },
                message=(
                    f"Final refit ended for best trial {best_trial_number}. "
                    f"trainer_log={final_refit_log_file}."
                ),
            )

        self._fit_posthoc_modules_from_oof(
            final_trainer.model,
            oof_logits=best_trial_result.aggregate_oof_logits,
            oof_targets=best_trial_result.aggregate_oof_targets,
        )

        holdout_metrics = None
        holdout_report_results = None
        if holdout_dataset is not None:
            holdout_metrics = self._evaluate_holdout(final_trainer, holdout_dataset)
            holdout_report_results = self._evaluate_report(final_trainer, holdout_dataset)

        final_model_state_dict_cpu = final_trainer._get_model_state_dict_cpu()
        final_model_state_dict_path = None

        if self.final_model_dir is not None and is_main_process:
            final_model_state_dict_path = os.path.join(
                self.final_model_dir,
                "final_model.pt",
            )
            torch.save(final_model_state_dict_cpu, final_model_state_dict_path)
            if run_logger is not None:
                run_logger.emit(
                    "cv_final_model_saved",
                    payload={"path": final_model_state_dict_path},
                    message=f"Final model state_dict saved to {final_model_state_dict_path}.",
                )

        if run_logger is not None:
            run_logger.emit(
                "cv_run_end",
                payload={
                    "best_trial_number": best_trial_number,
                    "best_metric": best_metric,
                    "best_selection_score": best_selection_score,
                    "attempted_trials": attempted_trials,
                    "successful_trials": successful_trials,
                    "failed_trials": failed_trials,
                    "pruned_trials": pruned_trials,
                    "holdout_metrics": copy.deepcopy(holdout_metrics),
                    "holdout_report_results": copy.deepcopy(holdout_report_results),
                },
                message=(
                    f"OptunaSearchCV ended. Best trial={best_trial_number}, "
                    f"best_metric={best_metric:.6f}, best_selection_score={best_selection_score:.6f}."
                ),
            )

        return OptunaSearchCVResult(
            search_pool_indices=copy.deepcopy(search_original_indices),
            trial_results=trial_results,
            best_params=best_params,
            best_metric=best_metric,
            best_selection_score=best_selection_score,
            best_trial_number=best_trial_number,
            attempted_trials=attempted_trials,
            successful_trials=successful_trials,
            failed_trials=failed_trials,
            pruned_trials=pruned_trials,
            selected_fold_results=copy.deepcopy(best_trial_result.fold_results),
            selected_fold_report_results=copy.deepcopy(best_trial_result.aggregate_fold_report_results),
            selected_metric_mean=selected_metric_mean,
            selected_metric_std=selected_metric_std,
            selected_metric_min=selected_metric_min,
            selected_metric_max=selected_metric_max,
            final_model_spec=copy.deepcopy(final_model_spec),
            final_trainer_spec=copy.deepcopy(final_trainer_spec),
            final_fit_epochs=final_fit_epochs,
            final_epochs_ran=int(final_trainer.state.epoch),
            final_best_epoch=final_trainer.state.best_epoch,
            final_best_metric=final_trainer.state.best_metric,
            final_train_logs=copy.deepcopy(final_trainer.state.train_logs),
            final_val_logs=copy.deepcopy(final_trainer.state.val_logs),
            final_history=copy.deepcopy(final_trainer.history),
            final_model_state_dict_cpu=final_model_state_dict_cpu if self.keep_final_model_state_dict_cpu else None,
            final_model_state_dict_path=final_model_state_dict_path,
            holdout_metrics=copy.deepcopy(holdout_metrics),
            holdout_report_results=copy.deepcopy(holdout_report_results),
            base_model_spec=copy.deepcopy(self.model_spec),
            base_trainer_spec=copy.deepcopy(self.trainer_spec),
            parameter_grid=copy.deepcopy(self.parameter_grid),
            report_evaluator=copy.deepcopy(self.report_evaluator),
            log_dir=self.log_dir,
            run_log_file=run_log_file,
            final_refit_log_file=final_refit_log_file,
            splitter_name=self.splitter_cls.__name__,
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
            n_trials=self.n_trials,
            max_trial_attempts=self.max_trial_attempts,
            calibrate=self.calibrate,
            final_model_dir=self.final_model_dir,
            keep_final_model_state_dict_cpu=self.keep_final_model_state_dict_cpu,
            selection_metric_name=selection_metric_name,
            selection_metric_direction=selection_metric_direction,
            selection_metric_spec=copy.deepcopy(selection_metric_spec),
        )

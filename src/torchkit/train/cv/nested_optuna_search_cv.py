from __future__ import annotations

from typing import Any, Optional

import copy
import json
import os
from pathlib import Path

import torch

from torchkit.data._dataset import DatasetSplit, TorchkitDataset
from torchkit.data.split import KFoldSplitter
from torchkit.evaluate.report.bundle import BundleReportEvaluator
from torchkit.train._event_log import JsonlEventLogger
from torchkit.train.cv._base_cv import _aggregate_report_results, _resolve_original_indices_for_subset
from torchkit.train.cv._base_search_cv import BaseSearchCV
from torchkit.train.cv._optuna_results import (
    NestedOptunaSearchCVResult,
    OptunaSearchCVResult,
    OuterFoldResult,
    _snapshot_object,
    _to_jsonable,
)
from torchkit.train.cv._optuna_search_mixin import ParameterGrid
from torchkit.train.cv.optuna_search_cv import OptunaSearchCV


class NestedOptunaSearchCV(BaseSearchCV):
    """
    Outer-loop orchestrator that composes the reusable OptunaSearchCV engine.

    Each outer fold runs one OptunaSearchCV on the outer-train subset.
    The resulting OptunaSearchCVResult is stored as a reusable container
    inside each OuterFoldResult.
    """

    def __init__(
        self,
        *,
        model_spec,
        trainer_spec,
        parameter_grid: ParameterGrid,
        outer_splitter_cls,
        inner_splitter_cls,
        dataloader_factory=None,
        n_trials: int = 10,
        max_trial_attempts: Optional[int] = None,
        k_outer: int = 5,
        k_inner: int = 3,
        shuffle_outer: bool = False,
        shuffle_inner: bool = False,
        random_state: Optional[int] = None,
        calibrate: bool = True,
        report_evaluator: Optional[BundleReportEvaluator] = None,
        logging: bool = False,
        _log_root_dir: Optional[str] = None,
        final_model_dir: Optional[str] = None,
        keep_final_model_state_dict_cpu: bool = True,
        posthoc_hooks: Optional[list] = None,
    ):
        super().__init__(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            parameter_grid=parameter_grid,
            splitter_cls=outer_splitter_cls,
            dataloader_factory=dataloader_factory,
            n_trials=n_trials,
            max_trial_attempts=max_trial_attempts,
            n_splits=k_outer,
            shuffle=shuffle_outer,
            random_state=random_state,
            calibrate=calibrate,
            report_evaluator=report_evaluator,
            logging=logging,
            _log_root_dir=_log_root_dir,
            final_model_dir=final_model_dir,
            keep_final_model_state_dict_cpu=keep_final_model_state_dict_cpu,
            posthoc_hooks=posthoc_hooks,
        )
        self.outer_splitter_cls: type[KFoldSplitter] = outer_splitter_cls
        self.inner_splitter_cls: type[KFoldSplitter] = inner_splitter_cls
        self.k_outer = int(k_outer)
        self.k_inner = int(k_inner)
        self.shuffle_outer = bool(shuffle_outer)
        self.shuffle_inner = bool(shuffle_inner)

        self.outer_splitter = outer_splitter_cls(
            n_splits=self.k_outer,
            shuffle=self.shuffle_outer,
            random_state=self.random_state,
        )

    def _outer_fold_model_dir(self, outer_fold: int) -> Optional[str]:
        if self.final_model_dir is None:
            return None
        path = os.path.join(self.final_model_dir, f"outer_fold_{outer_fold}")
        os.makedirs(path, exist_ok=True)
        return path

    def _build_inner_search(self, *, outer_fold: int) -> OptunaSearchCV:
        return OptunaSearchCV(
            model_spec=copy.deepcopy(self.model_spec),
            trainer_spec=copy.deepcopy(self.trainer_spec),
            parameter_grid=copy.deepcopy(self.parameter_grid),
            splitter_cls=self.inner_splitter_cls,
            dataloader_factory=self.dataloader_factory,
            n_trials=self.n_trials,
            max_trial_attempts=self.max_trial_attempts,
            n_splits=self.k_inner if self.k_inner is not None else 0,
            shuffle=self.shuffle_inner,
            random_state=self.random_state,
            calibrate=self.calibrate,
            report_evaluator=copy.deepcopy(self.report_evaluator),
            logging=self.logging,
            _log_root_dir=(
                os.path.join(self.log_dir, "outer_folds", f"outer_fold_{outer_fold:03d}", "inner_search")
                if self.logging and self.log_dir is not None
                else None
            ),
            final_model_dir=self._outer_fold_model_dir(outer_fold),
            keep_final_model_state_dict_cpu=self.keep_final_model_state_dict_cpu,
            posthoc_hooks=copy.deepcopy(self.posthoc_hooks),
        )

    @staticmethod
    def _write_json(path: str, payload: dict[str, Any]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _persist_outer_fold_artifacts(
        self,
        *,
        outer_fold: int,
        inner_search_result: OptunaSearchCVResult,
        outer_train_indices: list[int],
        outer_test_indices: list[int],
    ) -> dict[str, Optional[str]]:
        if not self._is_main_process():
            return {
                "selected_trial_summary_path": None,
                "refit_summary_path": None,
            }

        selected_summary_path = None
        refit_summary_path = None
        outer_fold_result_path = None

        outer_fold_log_dir = (
            os.path.join(self.log_dir, "outer_folds", f"outer_fold_{outer_fold:03d}")
            if self.log_dir is not None
            else None
        )
        outer_fold_model_dir = self._outer_fold_model_dir(outer_fold)

        selected_trial = inner_search_result.selected_trial_result()
        selected_fold_artifacts: list[dict[str, Any]] = []

        selected_ckpt_dir = None
        if outer_fold_model_dir is not None:
            selected_ckpt_dir = os.path.join(outer_fold_model_dir, "selected_trial")
            os.makedirs(selected_ckpt_dir, exist_ok=True)

        for fold_result in selected_trial.fold_results:
            checkpoint_path = None
            if selected_ckpt_dir is not None and fold_result.best_state_dict_cpu is not None:
                checkpoint_path = os.path.join(
                    selected_ckpt_dir,
                    f"trial_{selected_trial.trial_number:03d}_fold_{fold_result.fold:03d}_best_model.pt",
                )
                torch.save(fold_result.best_state_dict_cpu, checkpoint_path)

            selected_fold_artifacts.append(
                {
                    "fold": fold_result.fold,
                    "train_indices": copy.deepcopy(fold_result.train_indices),
                    "val_indices": copy.deepcopy(fold_result.val_indices),
                    "best_metric": fold_result.best_metric,
                    "best_epoch": fold_result.best_epoch,
                    "report_results": copy.deepcopy(fold_result.report_results),
                    "trainer_log_file": fold_result.log_file,
                    "checkpoint_path": checkpoint_path,
                }
            )

        selected_summary = {
            "artifact_type": "selected_trial_summary",
            "outer_fold": outer_fold,
            "status": "outer_fold_completed",
            "best_trial_number": inner_search_result.best_trial_number,
            "selected_trial_number": inner_search_result.best_trial_number,
            "best_params": copy.deepcopy(inner_search_result.best_params),
            "best_metric": inner_search_result.best_metric,
            "selected_trial_best_value": inner_search_result.best_metric,
            "best_selection_score": inner_search_result.best_selection_score,
            "selected_metric_mean": inner_search_result.selected_metric_mean,
            "selected_metric_std": inner_search_result.selected_metric_std,
            "selected_metric_min": inner_search_result.selected_metric_min,
            "selected_metric_max": inner_search_result.selected_metric_max,
            "selection_diagnostics": copy.deepcopy(inner_search_result.selection_diagnostics),
            "selected_competence_summary": copy.deepcopy(inner_search_result.selected_competence_summary),
            "selected_trial_main_metric": (
                None
                if inner_search_result.selection_diagnostics is None
                else copy.deepcopy(inner_search_result.selection_diagnostics.get("selected_trial_main_metric"))
            ),
            "selected_trial_probe_auc": (
                None
                if inner_search_result.selection_diagnostics is None
                else copy.deepcopy(inner_search_result.selection_diagnostics.get("selected_trial_probe_auc"))
            ),
            "selected_trial_probe_auc_defined": (
                None
                if inner_search_result.selection_diagnostics is None
                else copy.deepcopy(inner_search_result.selection_diagnostics.get("selected_trial_probe_auc_defined"))
            ),
            "search_pool_indices": copy.deepcopy(inner_search_result.search_pool_indices),
            "outer_train_indices": copy.deepcopy(outer_train_indices),
            "outer_test_indices": copy.deepcopy(outer_test_indices),
            "trial_log_file": selected_trial.log_file,
            "search_log_file": inner_search_result.run_log_file,
            "selected_fold_report_results": copy.deepcopy(inner_search_result.selected_fold_report_results),
            "selected_fold_report_results_raw": copy.deepcopy(
                inner_search_result.selected_fold_report_results_raw
            ),
            "fold_best_epochs": copy.deepcopy(inner_search_result.fold_best_epochs),
            "selected_folds": selected_fold_artifacts,
            "base_model_spec": _snapshot_object(copy.deepcopy(inner_search_result.base_model_spec)),
            "base_trainer_spec": _snapshot_object(copy.deepcopy(inner_search_result.base_trainer_spec)),
            "parameter_grid": copy.deepcopy(inner_search_result.parameter_grid),
            "selection_metric_name": inner_search_result.selection_metric_name,
            "selection_metric_direction": inner_search_result.selection_metric_direction,
            "selection_metric_spec": copy.deepcopy(inner_search_result.selection_metric_spec),
        }

        refit_summary = {
            "artifact_type": "refit_summary",
            "outer_fold": outer_fold,
            "status": "outer_fold_completed",
            "best_trial_number": inner_search_result.best_trial_number,
            "best_params": copy.deepcopy(inner_search_result.best_params),
            "final_model_state_dict_path": inner_search_result.final_model_state_dict_path,
            "final_refit_log_file": inner_search_result.final_refit_log_file,
            "search_log_file": inner_search_result.run_log_file,
            "outer_train_indices": copy.deepcopy(outer_train_indices),
            "outer_test_indices": copy.deepcopy(outer_test_indices),
            "fold_best_epochs": copy.deepcopy(inner_search_result.fold_best_epochs),
            "final_fit_epochs": inner_search_result.final_fit_epochs,
            "final_epochs_ran": inner_search_result.final_epochs_ran,
            "final_best_epoch": inner_search_result.final_best_epoch,
            "final_best_metric": inner_search_result.final_best_metric,
            "holdout_metrics": copy.deepcopy(inner_search_result.holdout_metrics),
            "holdout_metrics_by_phase": copy.deepcopy(inner_search_result.holdout_metrics_by_phase),
            "holdout_report_results": copy.deepcopy(inner_search_result.holdout_report_results),
            "holdout_report_results_by_phase": copy.deepcopy(inner_search_result.holdout_report_results_by_phase),
            "holdout_posthoc_results": copy.deepcopy(inner_search_result.holdout_posthoc_results),
            "final_posthoc_module_summary": copy.deepcopy(inner_search_result.final_posthoc_module_summary),
            "final_model_spec": _snapshot_object(copy.deepcopy(inner_search_result.final_model_spec)),
            "final_trainer_spec": _snapshot_object(copy.deepcopy(inner_search_result.final_trainer_spec)),
            "base_model_spec": _snapshot_object(copy.deepcopy(inner_search_result.base_model_spec)),
            "base_trainer_spec": _snapshot_object(copy.deepcopy(inner_search_result.base_trainer_spec)),
        }

        if outer_fold_log_dir is not None:
            selected_summary_path = os.path.join(outer_fold_log_dir, "selected_trial_summary.json")
            refit_summary_path = os.path.join(outer_fold_log_dir, "refit_summary.json")
            outer_fold_result_path = os.path.join(outer_fold_log_dir, "outer_fold_result.json")
            self._write_json(selected_summary_path, selected_summary)
            self._write_json(refit_summary_path, refit_summary)
            self._write_json(
                outer_fold_result_path,
                OuterFoldResult(
                    fold=outer_fold,
                    outer_train_indices=copy.deepcopy(outer_train_indices),
                    outer_test_indices=copy.deepcopy(outer_test_indices),
                    inner_search_result=copy.deepcopy(inner_search_result),
                    outer_test_metrics=copy.deepcopy(inner_search_result.holdout_metrics),
                    outer_test_metrics_by_phase=copy.deepcopy(inner_search_result.holdout_metrics_by_phase),
                    outer_test_report_results=copy.deepcopy(inner_search_result.holdout_report_results),
                    outer_test_report_results_by_phase=copy.deepcopy(inner_search_result.holdout_report_results_by_phase),
                    outer_test_posthoc_results=copy.deepcopy(inner_search_result.holdout_posthoc_results),
                    outer_test_posthoc_module_summary=copy.deepcopy(inner_search_result.final_posthoc_module_summary),
                    log_file=os.path.join(outer_fold_log_dir, f"outer_fold_{outer_fold:03d}.log.jsonl"),
                ).to_dict(include_tensors=False, include_tracebacks=True, include_inner_search_result=True),
            )

        return {
            "selected_trial_summary_path": selected_summary_path,
            "refit_summary_path": refit_summary_path,
            "outer_fold_result_path": outer_fold_result_path,
        }

    def _is_main_process(self) -> bool:
        strategy = getattr(self.trainer_spec, "distributed_strategy", None)
        if strategy is None or not strategy.is_enabled:
            return True
        return strategy.is_main_process

    def run(
        self,
        dataset: TorchkitDataset,
        index: Any = None,
        groups: Optional[Any] = None,
        *,
        outer_fold_indices: Optional[set[int]] = None,
    ) -> NestedOptunaSearchCVResult:
        strategy = getattr(self.trainer_spec, "distributed_strategy", None)
        if strategy is not None:
            strategy.initialize()

        try:
            outer_results: list[OuterFoldResult] = []
            run_logger = None
            run_log_file = None
            if self.logging and self.log_dir is not None and self._is_main_process():
                run_log_file = os.path.join(self.log_dir, "nested_search.log.jsonl")
                run_logger = JsonlEventLogger(
                    run_log_file,
                    scope="nested_optuna_search_cv",
                    echo_console=True,
                )
                run_logger.emit(
                    "nested_cv_run_start",
                    payload={
                        "k_outer": self.k_outer,
                        "k_inner": self.k_inner,
                        "n_trials": self.n_trials,
                        "outer_splitter_name": self.outer_splitter_cls.__name__,
                        "inner_splitter_name": self.inner_splitter_cls.__name__,
                        "dataset_size": len(dataset),
                        "log_dir": self.log_dir,
                    },
                    message=(
                        f"NestedOptunaSearchCV started: k_outer={self.k_outer}, k_inner={self.k_inner}, "
                        f"n_trials={self.n_trials}. Logging to {run_log_file}."
                    ),
                )

            selection_metric_name = self._selection_metric_name()
            selection_metric_direction = self._selection_metric_direction()
            selection_metric_spec = self._selection_metric_spec()

            requested_outer_folds = None if outer_fold_indices is None else {int(fold) for fold in outer_fold_indices}

            for outer_fold, (outer_train_subset, outer_test_subset) in enumerate(
                self._split(self.outer_splitter, dataset, index, groups)
            ):
                if requested_outer_folds is not None and outer_fold not in requested_outer_folds:
                    continue

                if strategy is not None:
                    strategy.barrier()

                outer_train_indices = _resolve_original_indices_for_subset(outer_train_subset)
                outer_test_indices = _resolve_original_indices_for_subset(outer_test_subset)
                outer_train_dataset = dataset.subset(outer_train_indices, split=DatasetSplit.TRAIN)
                outer_test_dataset = dataset.subset(outer_test_indices, split=DatasetSplit.TEST)
                outer_log_file = None
                outer_logger = None
                if self.logging and self.log_dir is not None and self._is_main_process():
                    outer_log_file = os.path.join(self.log_dir, "outer_folds", f"outer_fold_{outer_fold:03d}.log.jsonl")
                    outer_logger = JsonlEventLogger(
                        outer_log_file,
                        scope="nested_outer_fold",
                        echo_console=True,
                        context={"outer_fold": outer_fold},
                    )
                    outer_logger.emit(
                        "nested_outer_fold_start",
                        payload={
                            "outer_fold": outer_fold,
                            "n_outer_train": len(outer_train_indices),
                            "n_outer_test": len(outer_test_indices),
                        },
                        message=(
                            f"Outer fold {outer_fold} started "
                            f"(n_outer_train={len(outer_train_indices)}, n_outer_test={len(outer_test_indices)}). "
                            f"Logging to {outer_log_file}."
                        ),
                    )

                inner_search = self._build_inner_search(outer_fold=outer_fold)

                inner_search_result = inner_search.run(
                    outer_train_dataset,
                    index=index,
                    groups=groups,
                    holdout_dataset=outer_test_dataset,
                )
                artifact_paths = self._persist_outer_fold_artifacts(
                    outer_fold=outer_fold,
                    inner_search_result=inner_search_result,
                    outer_train_indices=outer_train_indices,
                    outer_test_indices=outer_test_indices,
                )

                if strategy is not None:
                    strategy.barrier()

                outer_results.append(
                    OuterFoldResult(
                        fold=outer_fold,
                        outer_train_indices=copy.deepcopy(outer_train_indices),
                        outer_test_indices=copy.deepcopy(outer_test_indices),
                        inner_search_result=inner_search_result,
                        outer_test_metrics=copy.deepcopy(inner_search_result.holdout_metrics),
                        outer_test_metrics_by_phase=copy.deepcopy(inner_search_result.holdout_metrics_by_phase),
                        outer_test_report_results=copy.deepcopy(inner_search_result.holdout_report_results),
                        outer_test_report_results_by_phase=copy.deepcopy(inner_search_result.holdout_report_results_by_phase),
                        outer_test_posthoc_results=copy.deepcopy(inner_search_result.holdout_posthoc_results),
                        outer_test_posthoc_module_summary=copy.deepcopy(inner_search_result.final_posthoc_module_summary),
                        log_file=outer_log_file,
                    )
                )
                if outer_logger is not None:
                    outer_logger.emit(
                        "nested_outer_fold_end",
                        payload={
                            "outer_fold": outer_fold,
                            "best_trial_number": inner_search_result.best_trial_number,
                            "best_metric": inner_search_result.best_metric,
                            "best_selection_score": inner_search_result.best_selection_score,
                            "outer_test_metrics": copy.deepcopy(inner_search_result.holdout_metrics),
                            "outer_test_metrics_by_phase": copy.deepcopy(inner_search_result.holdout_metrics_by_phase),
                            "outer_test_report_results": copy.deepcopy(inner_search_result.holdout_report_results),
                            "outer_test_report_results_by_phase": copy.deepcopy(inner_search_result.holdout_report_results_by_phase),
                            "outer_test_posthoc_results": copy.deepcopy(inner_search_result.holdout_posthoc_results),
                            "outer_test_posthoc_module_summary": copy.deepcopy(inner_search_result.final_posthoc_module_summary),
                            "inner_log_dir": inner_search_result.log_dir,
                            "selected_trial_summary_path": artifact_paths["selected_trial_summary_path"],
                            "refit_summary_path": artifact_paths["refit_summary_path"],
                            "outer_fold_result_path": artifact_paths["outer_fold_result_path"],
                            "final_model_state_dict_path": inner_search_result.final_model_state_dict_path,
                        },
                        message=(
                            f"Outer fold {outer_fold} ended. "
                            f"best_trial={inner_search_result.best_trial_number}, "
                            f"best_metric={inner_search_result.best_metric}, "
                            f"inner_log_dir={inner_search_result.log_dir}."
                        ),
                    )

            if requested_outer_folds is not None:
                completed_outer_folds = {int(result.fold) for result in outer_results}
                missing_outer_folds = sorted(requested_outer_folds - completed_outer_folds)
                if missing_outer_folds:
                    raise ValueError(
                        f"Requested outer folds were not produced: {missing_outer_folds}."
                    )

            outer_report_results = _aggregate_report_results(
                [outer.outer_test_report_results for outer in outer_results]
            )
            outer_posthoc_results = _aggregate_report_results(
                [outer.outer_test_posthoc_results for outer in outer_results]
            )
            if run_logger is not None:
                run_logger.emit(
                    "nested_cv_run_end",
                    payload={
                        "n_outer_folds": len(outer_results),
                        "outer_report_results": copy.deepcopy(outer_report_results),
                        "outer_posthoc_results": copy.deepcopy(outer_posthoc_results),
                    },
                    message=f"NestedOptunaSearchCV ended after {len(outer_results)} outer folds.",
                )

            return NestedOptunaSearchCVResult(
                outer_results=outer_results,
                base_model_spec=copy.deepcopy(self.model_spec),
                base_trainer_spec=copy.deepcopy(self.trainer_spec),
                parameter_grid=copy.deepcopy(self.parameter_grid),
                report_evaluator=copy.deepcopy(self.report_evaluator),
                outer_report_results=copy.deepcopy(outer_report_results),
                outer_posthoc_results=copy.deepcopy(outer_posthoc_results),
                posthoc_hooks=copy.deepcopy(self.posthoc_hooks),
                log_dir=self.log_dir,
                run_log_file=run_log_file,
                outer_splitter_name=self.outer_splitter_cls.__name__,
                inner_splitter_name=self.inner_splitter_cls.__name__ if self.inner_splitter_cls is not None else "",
                k_outer=self.k_outer,
                k_inner=self.k_inner if self.k_inner is not None else 0,
                shuffle_outer=self.shuffle_outer,
                shuffle_inner=self.shuffle_inner,
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
        finally:
            if strategy is not None:
                strategy.finalize()

    @staticmethod
    def aggregate_outer_results(
        outer_results: list[OuterFoldResult],
        *,
        model_spec: Any,
        trainer_spec: Any,
        parameter_grid: Any,
        report_evaluator: Any,
        posthoc_hooks: Any,
        log_dir: Optional[str],
        run_log_file: Optional[str],
        outer_splitter_name: str,
        inner_splitter_name: str,
        k_outer: int,
        k_inner: int,
        shuffle_outer: bool,
        shuffle_inner: bool,
        random_state: Optional[int],
        n_trials: int,
        max_trial_attempts: int,
        calibrate: bool,
        final_model_dir: Optional[str],
        keep_final_model_state_dict_cpu: bool,
        selection_metric_name: str,
        selection_metric_direction: Any,
        selection_metric_spec: dict[str, Any],
    ) -> NestedOptunaSearchCVResult:
        ordered_outer_results = sorted(copy.deepcopy(outer_results), key=lambda result: int(result.fold))
        outer_report_results = _aggregate_report_results(
            [outer.outer_test_report_results for outer in ordered_outer_results]
        )
        outer_posthoc_results = _aggregate_report_results(
            [outer.outer_test_posthoc_results for outer in ordered_outer_results]
        )
        return NestedOptunaSearchCVResult(
            outer_results=ordered_outer_results,
            base_model_spec=copy.deepcopy(model_spec),
            base_trainer_spec=copy.deepcopy(trainer_spec),
            parameter_grid=copy.deepcopy(parameter_grid),
            report_evaluator=copy.deepcopy(report_evaluator),
            outer_report_results=copy.deepcopy(outer_report_results),
            outer_posthoc_results=copy.deepcopy(outer_posthoc_results),
            posthoc_hooks=copy.deepcopy(posthoc_hooks),
            log_dir=log_dir,
            run_log_file=run_log_file,
            outer_splitter_name=outer_splitter_name,
            inner_splitter_name=inner_splitter_name,
            k_outer=k_outer,
            k_inner=k_inner,
            shuffle_outer=shuffle_outer,
            shuffle_inner=shuffle_inner,
            random_state=random_state,
            n_trials=n_trials,
            max_trial_attempts=max_trial_attempts,
            calibrate=calibrate,
            final_model_dir=final_model_dir,
            keep_final_model_state_dict_cpu=keep_final_model_state_dict_cpu,
            selection_metric_name=selection_metric_name,
            selection_metric_direction=selection_metric_direction,
            selection_metric_spec=copy.deepcopy(selection_metric_spec),
        )

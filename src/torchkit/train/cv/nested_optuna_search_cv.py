from __future__ import annotations

from typing import Any, Optional

import copy
import os

from torchkit.data._dataset import DatasetSplit, TorchkitDataset
from torchkit.data.split import KFoldSplitter
from torchkit.evaluate.report.bundle import BundleReportEvaluator
from torchkit.train._event_log import JsonlEventLogger
from torchkit.train.cv._base_cv import _aggregate_report_results, _resolve_original_indices_for_subset
from torchkit.train.cv._base_search_cv import BaseSearchCV
from torchkit.train.cv._optuna_results import (
    NestedOptunaSearchCVResult,
    OuterFoldResult,
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
    ) -> NestedOptunaSearchCVResult:
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

        for outer_fold, (outer_train_subset, outer_test_subset) in enumerate(
            self._split(self.outer_splitter, dataset, index, groups)
        ):
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

            outer_results.append(
                OuterFoldResult(
                    fold=outer_fold,
                    outer_train_indices=copy.deepcopy(outer_train_indices),
                    outer_test_indices=copy.deepcopy(outer_test_indices),
                    inner_search_result=inner_search_result,
                    outer_test_metrics=copy.deepcopy(inner_search_result.holdout_metrics),
                    outer_test_report_results=copy.deepcopy(inner_search_result.holdout_report_results),
                    outer_test_posthoc_results=copy.deepcopy(inner_search_result.holdout_posthoc_results),
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
                        "outer_test_report_results": copy.deepcopy(inner_search_result.holdout_report_results),
                        "outer_test_posthoc_results": copy.deepcopy(inner_search_result.holdout_posthoc_results),
                        "inner_log_dir": inner_search_result.log_dir,
                    },
                    message=(
                        f"Outer fold {outer_fold} ended. "
                        f"best_trial={inner_search_result.best_trial_number}, "
                        f"best_metric={inner_search_result.best_metric}, "
                        f"inner_log_dir={inner_search_result.log_dir}."
                    ),
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

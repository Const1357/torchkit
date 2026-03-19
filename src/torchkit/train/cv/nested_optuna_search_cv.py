from __future__ import annotations

from typing import Any, Optional

import copy
import os

from torch.utils.data import Subset

from torchkit.data._dataset import TorchkitDataset
from torchkit.train.cv._base_cv import _resolve_original_indices_for_subset
from torchkit.train.cv._base_search_cv import BaseSearchCV
from torchkit.train.cv._optuna_results import (
    NestedOptunaSearchCVResult,
    OuterFoldResult,
)
from torchkit.train.cv._optuna_search_mixin import SuggestionType
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
        parameter_grid: dict[str, tuple[list, SuggestionType]],
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
        final_model_dir: Optional[str] = None,
        keep_final_model_state_dict_cpu: bool = True,
    ):
        super().__init__(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            parameter_grid=parameter_grid,
            outer_splitter_cls=outer_splitter_cls,
            inner_splitter_cls=inner_splitter_cls,
            dataloader_factory=dataloader_factory,
            n_trials=n_trials,
            max_trial_attempts=max_trial_attempts,
            k_outer=k_outer,
            k_inner=k_inner,
            shuffle_outer=shuffle_outer,
            shuffle_inner=shuffle_inner,
            random_state=random_state,
            calibrate=calibrate,
            final_model_dir=final_model_dir,
            keep_final_model_state_dict_cpu=keep_final_model_state_dict_cpu,
        )

    def _outer_fold_model_dir(self, outer_fold: int) -> Optional[str]:
        if self.final_model_dir is None:
            return None
        path = os.path.join(self.final_model_dir, f"outer_fold_{outer_fold}")
        os.makedirs(path, exist_ok=True)
        return path

    def run(
        self,
        dataset: TorchkitDataset,
        index: Any = None,
        groups: Optional[Any] = None,
    ) -> NestedOptunaSearchCVResult:
        outer_results: list[OuterFoldResult] = []

        selection_metric_name = self._selection_metric_name()
        selection_metric_direction = self._selection_metric_direction()
        selection_metric_spec = self._selection_metric_spec()

        for outer_fold, (outer_train_subset, outer_test_subset) in enumerate(
            self._split(self.outer_splitter, dataset, index, groups)
        ):
            if not isinstance(outer_train_subset, Subset) or not isinstance(outer_test_subset, Subset):
                raise TypeError(
                    "KFoldSplitter wrappers are expected to return (Subset, Subset). "
                    f"Got ({type(outer_train_subset).__name__}, {type(outer_test_subset).__name__})."
                )

            outer_train_indices = _resolve_original_indices_for_subset(outer_train_subset)
            outer_test_indices = _resolve_original_indices_for_subset(outer_test_subset)

            inner_search = OptunaSearchCV(
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
                final_model_dir=self._outer_fold_model_dir(outer_fold),
                keep_final_model_state_dict_cpu=self.keep_final_model_state_dict_cpu,
            )

            inner_search_result = inner_search.run(
                outer_train_subset,
                index=index,
                groups=groups,
                holdout_dataset=outer_test_subset,
            )

            outer_results.append(
                OuterFoldResult(
                    fold=outer_fold,
                    outer_train_indices=copy.deepcopy(outer_train_indices),
                    outer_test_indices=copy.deepcopy(outer_test_indices),
                    inner_search_result=inner_search_result,
                    outer_test_metrics=copy.deepcopy(inner_search_result.holdout_metrics),
                )
            )

        return NestedOptunaSearchCVResult(
            outer_results=outer_results,
            base_model_spec=copy.deepcopy(self.model_spec),
            base_trainer_spec=copy.deepcopy(self.trainer_spec),
            parameter_grid=copy.deepcopy(self.parameter_grid),
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
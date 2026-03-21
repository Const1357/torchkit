from __future__ import annotations

from typing import Any, Callable, Optional

import copy

from torch.utils.data import DataLoader, Dataset

from torchkit.data.split import KFoldSplitter
from torchkit.evaluate.report.bundle import BundleReportEvaluator
from torchkit.models.Model._model import TorchkitModel
from torchkit.models.Model.factory import TorchkitModelSpec
from torchkit.train.cv._base_cv import BaseCV
from torchkit.train.cv._optuna_search_mixin import (
    ParameterGridLike,
    coerce_parameter_grid,
)
from torchkit.train.factory import TrainerFactory, TrainerSpec
from torchkit.train.trainer import Trainer


def _set_by_path(root: Any, path: str, value: Any) -> None:
    parts = [p for p in path.split("/") if p]
    if not parts:
        raise ValueError("Path must be non-empty.")

    cur = root
    for p in parts[:-1]:
        if isinstance(cur, dict):
            if p not in cur:
                raise KeyError(f"Key {p!r} not found while resolving path {path!r}.")
            cur = cur[p]
        else:
            if not hasattr(cur, p):
                raise AttributeError(f"Attribute {p!r} not found while resolving path {path!r}.")
            cur = getattr(cur, p)

    last = parts[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        if not hasattr(cur, last):
            raise AttributeError(f"Attribute {last!r} not found while resolving path {path!r}.")
        setattr(cur, last, value)


class BaseSearchCV(BaseCV):
    """
    Base class for search-based CV runners.

    This class is search-oriented but backend-agnostic:
    it knows about parameter grids and how to route sampled params into
    `model_spec` / `trainer_spec`, but does not assume Optuna.
    """

    def __init__(
        self,
        *,
        model_spec: TorchkitModelSpec | TorchkitModel,
        trainer_spec: TrainerSpec,
        parameter_grid: ParameterGridLike,
        splitter_cls: type[KFoldSplitter],
        dataloader_factory: Optional[Callable[[Dataset, bool], DataLoader]] = None,
        n_trials: int = 10,
        max_trial_attempts: Optional[int] = None,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        calibrate: bool = True,
        report_evaluator: Optional[BundleReportEvaluator] = None,
        final_model_dir: Optional[str] = None,
        keep_final_model_state_dict_cpu: bool = True,
    ):
        super().__init__(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            splitter_cls=splitter_cls,
            dataloader_factory=dataloader_factory,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            calibrate=calibrate,
            report_evaluator=report_evaluator,
            final_model_dir=final_model_dir,
            keep_final_model_state_dict_cpu=keep_final_model_state_dict_cpu,
        )

        self.parameter_grid = copy.deepcopy(coerce_parameter_grid(parameter_grid))
        self.n_trials = int(n_trials)

        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {self.n_trials}.")

        if max_trial_attempts is None:
            self.max_trial_attempts = max(5 * self.n_trials, self.n_trials)
        else:
            self.max_trial_attempts = int(max_trial_attempts)

        if self.max_trial_attempts < self.n_trials:
            raise ValueError(
                f"max_trial_attempts must be >= n_trials. "
                f"Got max_trial_attempts={self.max_trial_attempts}, n_trials={self.n_trials}."
            )

        self._validate_parameter_grid()

    def _validate_parameter_grid(self) -> None:
        """
        Base validation assumes entries are Optuna-style parameter specs that can be
        flattened into one or more concrete ``model/...`` or ``trainer/...`` assignments.
        """
        model_spec = copy.deepcopy(self.model_spec)
        trainer_spec = copy.deepcopy(self.trainer_spec)

        for path, spec in self.parameter_grid.suggestions.items():
            if not isinstance(path, str) or not path:
                raise ValueError(f"Invalid parameter path: {path!r}")

            self._validate_target_path(
                target_path=path,
                value=self._dummy_value_for_validation(spec.values),
                model_spec=model_spec,
                trainer_spec=trainer_spec,
            )

        sampled_values = {
            path: self._dummy_value_for_validation(spec.values)
            for path, spec in self.parameter_grid.suggestions.items()
        }
        for derived in self.parameter_grid.derived_params:
            arg_values = [sampled_values[arg] for arg in derived.args]
            self._validate_target_path(
                target_path=derived.target_path,
                value=derived.transform(*arg_values),
                model_spec=model_spec,
                trainer_spec=trainer_spec,
            )

    @staticmethod
    def _dummy_value_for_validation(values: list[Any]) -> Any:
        if len(values) == 0:
            raise ValueError("Parameter grid values list must be non-empty.")
        return values[0]

    @staticmethod
    def _validate_target_path(
        *,
        target_path: str,
        value: Any,
        model_spec: TorchkitModelSpec,
        trainer_spec: TrainerSpec,
    ) -> None:
        if target_path.startswith("model/"):
            target = model_spec
            rel_path = target_path.removeprefix("model/")
        elif target_path.startswith("trainer/"):
            target = trainer_spec
            rel_path = target_path.removeprefix("trainer/")
        else:
            raise ValueError(
                f"Parameter path {target_path!r} must start with 'model/' or 'trainer/'."
            )

        _set_by_path(target, rel_path, value)

    def _apply_suggested_params(
        self,
        *,
        model_spec: TorchkitModelSpec,
        trainer_spec: TrainerSpec,
        params: dict[str, Any],
    ) -> None:
        for path, value in params.items():
            if path.startswith("model/"):
                _set_by_path(model_spec, path.removeprefix("model/"), value)
            elif path.startswith("trainer/"):
                _set_by_path(trainer_spec, path.removeprefix("trainer/"), value)
            else:
                raise ValueError(
                    f"Parameter path {path!r} must start with 'model/' or 'trainer/'."
                )

    def _build_trainer_for_trial(
        self,
        *,
        params: dict[str, Any],
    ) -> tuple[TorchkitModelSpec, TrainerSpec, Trainer]:
        model_spec = copy.deepcopy(self.model_spec)
        trainer_spec = copy.deepcopy(self.trainer_spec)

        self._apply_suggested_params(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            params=params,
        )

        trainer = TrainerFactory.build_from_model_spec(
            trainer_spec,
            model_spec=model_spec,
        )
        return model_spec, trainer_spec, trainer

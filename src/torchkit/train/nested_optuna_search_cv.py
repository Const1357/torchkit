from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple

import copy
import os
import statistics
import traceback

import optuna
from optuna.trial import TrialState
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from torchkit.data._dataset import TorchkitDataset
from torchkit.data.split import KFoldSplitter
from torchkit.models.Model.factory import TorchkitModelFactory, TorchkitModelSpec
from torchkit.train.factory import TrainerFactory, TrainerSpec
from torchkit.train.trainer import Trainer


SuggestionType = Literal[
    "categorical",
    "float",
    "int",
    "loguniform",
    "uniform",
    "discrete_uniform",
]

TrialStatus = Literal[
    "SUCCESS",
    "FAILED",
    "PRUNED",
]

MetricDirection = Literal["maximize", "minimize"]


@dataclass
class InnerFoldResult:
    fold: int

    inner_train_indices: list[int] = field(default_factory=list)
    inner_val_indices: list[int] = field(default_factory=list)

    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    best_state_dict_cpu: Optional[dict[str, torch.Tensor]] = None

    oof_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    oof_targets: dict[str, torch.Tensor] = field(default_factory=dict)
    oof_sample_indices: list[int] = field(default_factory=list)


@dataclass
class TrialResult:
    trial_number: int
    params: dict[str, Any]
    status: TrialStatus

    aggregate_metric: Optional[float]
    aggregate_selection_score: Optional[float]

    inner_results: list[InnerFoldResult] = field(default_factory=list)

    aggregate_oof_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    aggregate_oof_targets: dict[str, torch.Tensor] = field(default_factory=dict)
    aggregate_oof_sample_indices: list[int] = field(default_factory=list)

    error_message: Optional[str] = None
    error_traceback: Optional[str] = None


@dataclass
class OuterFoldResult:
    fold: int

    outer_train_indices: list[int] = field(default_factory=list)
    outer_test_indices: list[int] = field(default_factory=list)

    best_params: dict[str, Any] = field(default_factory=dict)
    best_metric: float = 0.0
    best_selection_score: float = 0.0
    best_trial_number: int = -1

    attempted_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    pruned_trials: int = 0

    trial_results: list[TrialResult] = field(default_factory=list)

    selected_inner_results: list[InnerFoldResult] = field(default_factory=list)
    selected_inner_metric_mean: Optional[float] = None
    selected_inner_metric_std: Optional[float] = None
    selected_inner_metric_min: Optional[float] = None
    selected_inner_metric_max: Optional[float] = None

    final_model_spec: Optional[TorchkitModelSpec] = None
    final_trainer_spec: Optional[TrainerSpec] = None

    final_fit_epochs: Optional[int] = None
    final_epochs_ran: Optional[int] = None

    # For the final refit (which is train-only by design), these may legitimately be None.
    final_best_epoch: Optional[int] = None
    final_best_metric: Optional[float] = None

    final_train_logs: list[dict[str, Any]] = field(default_factory=list)
    final_val_logs: list[dict[str, Any]] = field(default_factory=list)
    final_history: list[dict[str, Any]] = field(default_factory=list)

    final_model_state_dict_cpu: Optional[dict[str, torch.Tensor]] = None
    final_model_state_dict_path: Optional[str] = None

    test_metrics: Optional[dict[str, Any]] = None


@dataclass
class NestedCVResult:
    outer_results: list[OuterFoldResult]

    # CV-level metadata for offline reporting / auditability
    base_model_spec: TorchkitModelSpec
    base_trainer_spec: TrainerSpec
    parameter_grid: dict[str, Tuple[list, SuggestionType]]

    outer_splitter_name: str
    inner_splitter_name: str
    k_outer: int
    k_inner: int
    shuffle_outer: bool
    shuffle_inner: bool
    random_state: Optional[int]

    n_trials: int
    max_trial_attempts: int
    calibrate: bool
    final_model_dir: Optional[str]
    keep_final_model_state_dict_cpu: bool

    selection_metric_name: str
    selection_metric_direction: MetricDirection

    def rebuild_final_model(
        self,
        outer_fold: int,
        *,
        device: torch.device | str = "cpu",
    ):
        outer = self.outer_results[outer_fold]

        if outer.final_model_spec is None:
            raise ValueError(f"Outer fold {outer_fold} does not contain final_model_spec.")

        if outer.final_model_state_dict_path is not None:
            return TorchkitModelFactory.build(
                copy.deepcopy(outer.final_model_spec),
                state_dict_path=outer.final_model_state_dict_path,
                device=device,
            )

        if outer.final_model_state_dict_cpu is not None:
            return TorchkitModelFactory.build(
                copy.deepcopy(outer.final_model_spec),
                state_dict=outer.final_model_state_dict_cpu,
                device=device,
            )

        raise ValueError(
            f"Outer fold {outer_fold} does not contain a saved final model state_dict "
            f"(neither in-memory nor on disk)."
        )

    def rebuild_final_trainer(
        self,
        outer_fold: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Trainer:
        outer = self.outer_results[outer_fold]

        if outer.final_trainer_spec is None:
            raise ValueError(f"Outer fold {outer_fold} does not contain final_trainer_spec.")

        model = self.rebuild_final_model(outer_fold, device=device)

        trainer_spec = copy.deepcopy(outer.final_trainer_spec)
        trainer_spec.config.device = device

        return TrainerFactory.build(
            trainer_spec,
            model=model,
        )


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


def _safe_take(values: Any, indices: list[int] | tuple[int, ...] | Any) -> Any:
    if values is None:
        return None

    if hasattr(values, "take"):
        try:
            return values.take(indices)
        except Exception:
            pass

    if hasattr(values, "iloc"):
        try:
            return values.iloc[list(indices)]
        except Exception:
            pass

    if torch.is_tensor(values):
        return values[list(indices)]

    return [values[i] for i in indices]


def _clone_tensor_dict(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in d.items()}


def _clone_state_dict_cpu(sd: Optional[dict[str, torch.Tensor]]) -> Optional[dict[str, torch.Tensor]]:
    if sd is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def _concat_tensor_dicts(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    buckets: dict[str, list[torch.Tensor]] = {}
    for d in dicts:
        for k, v in d.items():
            buckets.setdefault(k, []).append(v.detach().cpu())

    merged: dict[str, torch.Tensor] = {}
    for k, vs in buckets.items():
        if len(vs) == 1:
            merged[k] = vs[0].clone()
        else:
            merged[k] = torch.cat(vs, dim=0)
    return merged


def _resolve_original_indices_for_subset(subset: Subset) -> list[int]:
    """
    Resolve nested Subset indices back to original dataset coordinates.
    """
    indices = list(subset.indices)
    base = subset.dataset

    while isinstance(base, Subset):
        parent_indices = list(base.indices)
        indices = [parent_indices[i] for i in indices]
        base = base.dataset

    return indices


class NestedOptunaSearchCV:

    def __init__(
        self,
        *,
        model_spec: TorchkitModelSpec,
        trainer_spec: TrainerSpec,
        parameter_grid: dict[str, Tuple[list, SuggestionType]],
        outer_splitter_cls: type[KFoldSplitter],
        inner_splitter_cls: type[KFoldSplitter],
        dataloader_factory: Optional[Callable[[Dataset, bool], DataLoader]] = None,
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
        self.model_spec = copy.deepcopy(model_spec)
        self.trainer_spec = copy.deepcopy(trainer_spec)
        self.parameter_grid = copy.deepcopy(parameter_grid)

        self.outer_splitter_cls = outer_splitter_cls
        self.inner_splitter_cls = inner_splitter_cls

        self.k_outer = int(k_outer)
        self.k_inner = int(k_inner)
        self.shuffle_outer = bool(shuffle_outer)
        self.shuffle_inner = bool(shuffle_inner)
        self.random_state = random_state

        self.calibrate = calibrate
        self.n_trials = int(n_trials)
        self.final_model_dir = final_model_dir
        self.keep_final_model_state_dict_cpu = bool(keep_final_model_state_dict_cpu)

        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {self.n_trials}.")

        if max_trial_attempts is None:
            self.max_trial_attempts = max(5 * self.n_trials, self.n_trials)
        else:
            self.max_trial_attempts = int(max_trial_attempts)

        if self.max_trial_attempts < self.n_trials:
            raise ValueError(
                f"max_trial_attempts must be >= n_trials. Got max_trial_attempts={self.max_trial_attempts}, n_trials={self.n_trials}."
            )

        if self.final_model_dir is not None:
            os.makedirs(self.final_model_dir, exist_ok=True)

        if (self.final_model_dir is None) and (not self.keep_final_model_state_dict_cpu):
            raise ValueError(
                "Final models would be unrebuildable: both final_model_dir is None and "
                "keep_final_model_state_dict_cpu is False."
            )

        self.outer_splitter = outer_splitter_cls(
            n_splits=self.k_outer,
            shuffle=self.shuffle_outer,
            random_state=self.random_state,
        )

        self.inner_splitter = inner_splitter_cls(
            n_splits=self.k_inner,
            shuffle=self.shuffle_inner,
            random_state=self.random_state,
        )

        if dataloader_factory is None:
            self.dataloader_factory = lambda ds, shuffle: DataLoader(ds, batch_size=1, shuffle=shuffle)
        else:
            self.dataloader_factory = dataloader_factory

        self._validate_parameter_grid()

    def _selection_metric_name(self) -> str:
        dataset_evaluator = getattr(self.trainer_spec, "dataset_evaluator", None)
        if dataset_evaluator is not None:
            return str(dataset_evaluator.primary_metric)
        return "val_loss"

    def _selection_metric_direction(self) -> MetricDirection:
        dataset_evaluator = getattr(self.trainer_spec, "dataset_evaluator", None)
        if dataset_evaluator is not None:
            return str(dataset_evaluator.direction)  # type: ignore[return-value]
        return "minimize"

    def _to_selection_score(self, raw_metric: float) -> float:
        direction = self._selection_metric_direction()
        if direction == "maximize":
            return float(raw_metric)
        if direction == "minimize":
            return -float(raw_metric)
        raise ValueError(f"Unsupported selection metric direction {direction!r}.")

    def _validate_parameter_grid(self) -> None:
        model_spec = copy.deepcopy(self.model_spec)
        trainer_spec = copy.deepcopy(self.trainer_spec)

        for path, spec in self.parameter_grid.items():
            if not isinstance(path, str) or not path:
                raise ValueError(f"Invalid parameter path: {path!r}")

            if not isinstance(spec, tuple) or len(spec) != 2:
                raise ValueError(
                    f"Parameter grid entry for {path!r} must be a tuple of (values, suggestion_type)."
                )

            param_values, suggestion_type = spec

            if path.startswith("model/"):
                target = model_spec
                rel_path = path.removeprefix("model/")
            elif path.startswith("trainer/"):
                target = trainer_spec
                rel_path = path.removeprefix("trainer/")
            else:
                raise ValueError(
                    f"Parameter path {path!r} must start with 'model/' or 'trainer/'."
                )

            _set_by_path(target, rel_path, self._dummy_value_for_validation(param_values, suggestion_type))

            if suggestion_type == "categorical":
                if not isinstance(param_values, (list, tuple)) or len(param_values) == 0:
                    raise ValueError(f"{path!r}: categorical requires a non-empty list/tuple of values.")
            elif suggestion_type in ("float", "int", "loguniform", "uniform"):
                if not isinstance(param_values, (list, tuple)) or len(param_values) != 2:
                    raise ValueError(f"{path!r}: {suggestion_type} requires exactly 2 values.")
            elif suggestion_type == "discrete_uniform":
                if not isinstance(param_values, (list, tuple)) or len(param_values) != 3:
                    raise ValueError(f"{path!r}: discrete_uniform requires exactly 3 values: (low, high, q).")
            else:
                raise ValueError(f"{path!r}: unsupported suggestion_type {suggestion_type!r}.")

    @staticmethod
    def _dummy_value_for_validation(param_values: list, suggestion_type: SuggestionType) -> Any:
        if suggestion_type == "categorical":
            return param_values[0]
        return param_values[0]

    @staticmethod
    def suggest_parameters(
        trial: optuna.Trial,
        parameter_grid: dict[str, Tuple[list, SuggestionType]],
    ) -> dict[str, Any]:
        suggested_params: dict[str, Any] = {}

        for param_name, (param_values, suggestion_type) in parameter_grid.items():
            if suggestion_type == "categorical":
                suggested_params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif suggestion_type == "float":
                suggested_params[param_name] = trial.suggest_float(param_name, *param_values)
            elif suggestion_type == "int":
                suggested_params[param_name] = trial.suggest_int(param_name, *param_values)
            elif suggestion_type == "loguniform":
                suggested_params[param_name] = trial.suggest_float(param_name, *param_values, log=True)
            elif suggestion_type == "uniform":
                suggested_params[param_name] = trial.suggest_float(param_name, *param_values)
            elif suggestion_type == "discrete_uniform":
                low, high, q = param_values
                suggested_params[param_name] = trial.suggest_float(param_name, low, high, step=q)
            else:
                raise ValueError(f"Unsupported suggestion_type {suggestion_type!r} for parameter {param_name!r}.")

        return suggested_params

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
        self._apply_suggested_params(model_spec=model_spec, trainer_spec=trainer_spec, params=params)

        trainer = TrainerFactory.build_from_model_spec(
            trainer_spec,
            model_spec=model_spec,
        )
        return model_spec, trainer_spec, trainer
    
    def _split(
        self,
        splitter: KFoldSplitter,
        dataset: TorchkitDataset | Subset,
        y: Any,
        groups: Optional[Any] = None,
    ):
        if groups is None:
            return splitter.split(dataset, y)
        return splitter.split(dataset, y, groups)

    def _fit_trial_calibrators(self, model: Any, trial_result: TrialResult) -> None:
        if not self.calibrate:
            return

        prediction_heads = getattr(model, "prediction_heads", None)
        if prediction_heads is None:
            return

        for task, prediction_head in prediction_heads.items():
            if prediction_head is None or not getattr(prediction_head, "is_active", True):
                continue

            calibrator = getattr(prediction_head, "calibrator", None)
            if calibrator is None or not getattr(calibrator, "is_active", True):
                continue

            if task not in trial_result.aggregate_oof_logits or task not in trial_result.aggregate_oof_targets:
                raise ValueError(
                    f"Calibrator for task {task!r} is active, but aggregate OOF logits/targets are missing."
                )

            logits = trial_result.aggregate_oof_logits[task]
            targets = trial_result.aggregate_oof_targets[task]
            calibrator.fit(logits=logits, targets=targets)
            calibrator.enable()

    def _evaluate_holdout(
        self,
        trainer: Trainer,
        dataset_subset: Subset | Dataset,
    ) -> dict[str, Any]:
        loader = self.dataloader_factory(dataset_subset, False)
        state_backup = copy.deepcopy(trainer.state)
        try:
            metrics = trainer._validate_one_epoch(loader, epoch=0)
        finally:
            trainer.state = state_backup
        return metrics

    def _run_single_trial(
        self,
        *,
        trial: optuna.Trial,
        outer_train_dataset: Subset,
        outer_train_index: Any,
        outer_train_groups: Any,
        outer_train_original_indices: list[int],
    ) -> TrialResult:
        params = self.suggest_parameters(trial, self.parameter_grid)
        _, _, trainer = self._build_trainer_for_trial(params=params)

        inner_results: list[InnerFoldResult] = []
        inner_metrics: list[float] = []
        inner_selection_scores: list[float] = []

        inner_oof_logits_all: list[dict[str, torch.Tensor]] = []
        inner_oof_targets_all: list[dict[str, torch.Tensor]] = []
        aggregate_oof_sample_indices: list[int] = []

        for inner_fold, (inner_train_subset, inner_val_subset) in enumerate(
            self._split(self.inner_splitter, outer_train_dataset, outer_train_index, outer_train_groups)
        ):
            if not isinstance(inner_train_subset, Subset) or not isinstance(inner_val_subset, Subset):
                raise TypeError(
                    "KFoldSplitter wrappers are expected to return (Subset, Subset). "
                    f"Got ({type(inner_train_subset).__name__}, {type(inner_val_subset).__name__})."
                )

            inner_train_original_indices = _resolve_original_indices_for_subset(inner_train_subset)
            inner_val_original_indices = _resolve_original_indices_for_subset(inner_val_subset)

            train_loader = self.dataloader_factory(inner_train_subset, True)
            val_loader = self.dataloader_factory(inner_val_subset, False)

            trainer.reset_state()
            trainer.fit(
                train_loader,
                val_loader,
                trial=None,
            )

            metric = trainer.state.best_metric
            if metric is not None:
                metric = float(metric)

            fold_result = InnerFoldResult(
                fold=inner_fold,
                inner_train_indices=copy.deepcopy(inner_train_original_indices),
                inner_val_indices=copy.deepcopy(inner_val_original_indices),
                best_metric=metric,
                best_epoch=trainer.state.best_epoch,
                best_state_dict_cpu=_clone_state_dict_cpu(trainer.state.best_state_dict_cpu),
                oof_logits=_clone_tensor_dict(trainer.state.oof_logits),
                oof_targets=_clone_tensor_dict(trainer.state.oof_targets),
                oof_sample_indices=copy.deepcopy(inner_val_original_indices),
            )
            inner_results.append(fold_result)

            if metric is not None:
                inner_metrics.append(metric)
                inner_selection_scores.append(self._to_selection_score(metric))

            if trainer.state.oof_logits:
                inner_oof_logits_all.append(_clone_tensor_dict(trainer.state.oof_logits))
            if trainer.state.oof_targets:
                inner_oof_targets_all.append(_clone_tensor_dict(trainer.state.oof_targets))
            if trainer.state.oof_logits or trainer.state.oof_targets:
                aggregate_oof_sample_indices.extend(inner_val_original_indices)

        if len(inner_metrics) == 0:
            raise ValueError(f"Trial {trial.number} produced no valid inner-fold metrics.")

        aggregate_metric = sum(inner_metrics) / len(inner_metrics)
        aggregate_selection_score = sum(inner_selection_scores) / len(inner_selection_scores)

        aggregate_oof_logits = _concat_tensor_dicts(inner_oof_logits_all) if inner_oof_logits_all else {}
        aggregate_oof_targets = _concat_tensor_dicts(inner_oof_targets_all) if inner_oof_targets_all else {}

        # OOF auditability / leakage guard:
        # if OOF exists, it must cover each outer-train sample exactly once.
        if aggregate_oof_sample_indices:
            if len(aggregate_oof_sample_indices) != len(set(aggregate_oof_sample_indices)):
                raise ValueError(
                    f"Trial {trial.number} produced duplicated OOF sample indices. "
                    "This indicates leakage or overlapping inner validation folds."
                )

            if sorted(aggregate_oof_sample_indices) != sorted(outer_train_original_indices):
                raise ValueError(
                    f"Trial {trial.number} produced OOF sample indices that do not exactly cover outer-train. "
                    "This indicates missing or leaked samples in inner-fold OOF aggregation."
                )

        return TrialResult(
            trial_number=trial.number,
            params=copy.deepcopy(params),
            status="SUCCESS",
            aggregate_metric=float(aggregate_metric),
            aggregate_selection_score=float(aggregate_selection_score),
            inner_results=inner_results,
            aggregate_oof_logits=aggregate_oof_logits,
            aggregate_oof_targets=aggregate_oof_targets,
            aggregate_oof_sample_indices=copy.deepcopy(aggregate_oof_sample_indices),
            error_message=None,
            error_traceback=None,
        )

    def run(
        self,
        dataset: TorchkitDataset,
        index: Any = None,
        groups: Optional[Any] = None,
    ) -> NestedCVResult:
        outer_results: list[OuterFoldResult] = []

        selection_metric_name = self._selection_metric_name()
        selection_metric_direction = self._selection_metric_direction()

        for outer_fold, (outer_train_subset, outer_test_subset) in enumerate(
            self._split(self.outer_splitter, dataset, index, groups)
        ):
            if not isinstance(outer_train_subset, Subset) or not isinstance(outer_test_subset, Subset):
                raise TypeError(
                    "KFoldSplitter wrappers are expected to return (Subset, Subset). "
                    f"Got ({type(outer_train_subset).__name__}, {type(outer_test_subset).__name__})."
                )

            outer_train_dataset = outer_train_subset
            outer_test_dataset = outer_test_subset

            outer_train_original_indices = _resolve_original_indices_for_subset(outer_train_dataset)
            outer_test_original_indices = _resolve_original_indices_for_subset(outer_test_dataset)

            outer_train_index = _safe_take(index, outer_train_original_indices) if index is not None else None
            outer_train_groups = _safe_take(groups, outer_train_original_indices) if groups is not None else None

            trial_results: list[TrialResult] = []

            study = optuna.create_study(direction="maximize")

            attempted_trials = 0
            successful_trials = 0
            failed_trials = 0
            pruned_trials = 0

            while successful_trials < self.n_trials:
                if attempted_trials >= self.max_trial_attempts:
                    raise RuntimeError(
                        f"Reached max_trial_attempts={self.max_trial_attempts} on outer fold {outer_fold} "
                        f"before obtaining {self.n_trials} successful trials. "
                        f"Successful={successful_trials}, failed={failed_trials}, pruned={pruned_trials}."
                    )

                trial = study.ask()
                attempted_trials += 1

                try:
                    trial_result = self._run_single_trial(
                        trial=trial,
                        outer_train_dataset=outer_train_dataset,
                        outer_train_index=outer_train_index,
                        outer_train_groups=outer_train_groups,
                        outer_train_original_indices=outer_train_original_indices,
                    )
                    assert trial_result.aggregate_selection_score is not None
                    study.tell(trial, trial_result.aggregate_selection_score)
                    trial_results.append(trial_result)
                    successful_trials += 1

                except optuna.TrialPruned as e:
                    tb = traceback.format_exc()
                    study.tell(trial, state=TrialState.PRUNED)
                    trial_results.append(
                        TrialResult(
                            trial_number=trial.number,
                            params=copy.deepcopy(dict(trial.params)),
                            status="PRUNED",
                            aggregate_metric=None,
                            aggregate_selection_score=None,
                            inner_results=[],
                            aggregate_oof_logits={},
                            aggregate_oof_targets={},
                            aggregate_oof_sample_indices=[],
                            error_message=str(e),
                            error_traceback=tb,
                        )
                    )
                    pruned_trials += 1

                except Exception as e:
                    tb = traceback.format_exc()
                    study.tell(trial, state=TrialState.FAIL)
                    trial_results.append(
                        TrialResult(
                            trial_number=trial.number,
                            params=copy.deepcopy(dict(trial.params)),
                            status="FAILED",
                            aggregate_metric=None,
                            aggregate_selection_score=None,
                            inner_results=[],
                            aggregate_oof_logits={},
                            aggregate_oof_targets={},
                            aggregate_oof_sample_indices=[],
                            error_message=f"{type(e).__name__}: {e}",
                            error_traceback=tb,
                        )
                    )
                    failed_trials += 1

            successful_trial_results = [tr for tr in trial_results if tr.status == "SUCCESS"]
            if len(successful_trial_results) == 0:
                raise RuntimeError(f"Outer fold {outer_fold} produced no successful trials.")

            best_trial_number = study.best_trial.number

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

            selected_inner_metrics = [
                float(r.best_metric) for r in best_trial_result.inner_results if r.best_metric is not None
            ]
            selected_inner_metric_mean = (
                float(statistics.mean(selected_inner_metrics)) if selected_inner_metrics else None
            )
            selected_inner_metric_std = (
                float(statistics.stdev(selected_inner_metrics))
                if len(selected_inner_metrics) >= 2
                else 0.0 if len(selected_inner_metrics) == 1
                else None
            )
            selected_inner_metric_min = (
                float(min(selected_inner_metrics)) if selected_inner_metrics else None
            )
            selected_inner_metric_max = (
                float(max(selected_inner_metrics)) if selected_inner_metrics else None
            )

            final_model_spec, final_trainer_spec, final_trainer = self._build_trainer_for_trial(params=best_params)
            outer_train_loader = self.dataloader_factory(outer_train_dataset, True)

            inner_best_epochs = [r.best_epoch for r in best_trial_result.inner_results if r.best_epoch is not None]
            final_fit_epochs = (
                int(statistics.median(inner_best_epochs))
                if inner_best_epochs
                else int(final_trainer.config.max_epochs)
            )

            final_trainer.fit(
                outer_train_loader,
                val_loader=None,
                reset_state=True,
                max_epochs=final_fit_epochs,
                early_stopping_patience=None,
            )

            self._fit_trial_calibrators(final_trainer.model, best_trial_result)
            test_metrics = self._evaluate_holdout(final_trainer, outer_test_dataset)

            final_model_state_dict_cpu = final_trainer._get_model_state_dict_cpu()
            final_model_state_dict_path = None

            if self.final_model_dir is not None:
                final_model_state_dict_path = os.path.join(
                    self.final_model_dir,
                    f"final_model_fold{outer_fold}.pt",
                )
                torch.save(final_model_state_dict_cpu, final_model_state_dict_path)

            outer_results.append(
                OuterFoldResult(
                    fold=outer_fold,
                    outer_train_indices=copy.deepcopy(outer_train_original_indices),
                    outer_test_indices=copy.deepcopy(outer_test_original_indices),
                    best_params=copy.deepcopy(best_params),
                    best_metric=best_metric,
                    best_selection_score=best_selection_score,
                    best_trial_number=best_trial_number,
                    attempted_trials=attempted_trials,
                    successful_trials=successful_trials,
                    failed_trials=failed_trials,
                    pruned_trials=pruned_trials,
                    trial_results=trial_results,
                    selected_inner_results=copy.deepcopy(best_trial_result.inner_results),
                    selected_inner_metric_mean=selected_inner_metric_mean,
                    selected_inner_metric_std=selected_inner_metric_std,
                    selected_inner_metric_min=selected_inner_metric_min,
                    selected_inner_metric_max=selected_inner_metric_max,
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
                    test_metrics=copy.deepcopy(test_metrics),
                )
            )

        return NestedCVResult(
            outer_results=outer_results,
            base_model_spec=copy.deepcopy(self.model_spec),
            base_trainer_spec=copy.deepcopy(self.trainer_spec),
            parameter_grid=copy.deepcopy(self.parameter_grid),
            outer_splitter_name=self.outer_splitter_cls.__name__,
            inner_splitter_name=self.inner_splitter_cls.__name__,
            k_outer=self.k_outer,
            k_inner=self.k_inner,
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
        )
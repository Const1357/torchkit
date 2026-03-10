from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Tuple

import copy
import os
import statistics

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


@dataclass
class InnerFoldResult:
    fold: int
    best_metric: Optional[float]
    best_epoch: Optional[int]
    best_state_dict_cpu: Optional[dict[str, torch.Tensor]] = None
    oof_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    oof_targets: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_number: int
    params: dict[str, Any]
    status: TrialStatus
    aggregate_metric: Optional[float]
    inner_results: list[InnerFoldResult] = field(default_factory=list)
    aggregate_oof_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    aggregate_oof_targets: dict[str, torch.Tensor] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class OuterFoldResult:
    fold: int
    best_params: dict[str, Any]
    best_metric: float
    best_trial_number: int
    attempted_trials: int
    successful_trials: int
    failed_trials: int
    pruned_trials: int
    trial_results: list[TrialResult]

    selected_inner_results: list[InnerFoldResult] = field(default_factory=list)
    selected_inner_metric_mean: Optional[float] = None
    selected_inner_metric_std: Optional[float] = None
    selected_inner_metric_min: Optional[float] = None
    selected_inner_metric_max: Optional[float] = None

    final_model_spec: Optional[TorchkitModelSpec] = None
    final_trainer_spec: Optional[TrainerSpec] = None

    final_fit_epochs: Optional[int] = None
    final_best_epoch: Optional[int] = None
    final_best_metric: Optional[float] = None
    final_model_state_dict_cpu: Optional[dict[str, torch.Tensor]] = None
    final_model_state_dict_path: Optional[str] = None

    test_metrics: Optional[dict[str, Any]] = None


@dataclass
class NestedCVResult:
    outer_results: list[OuterFoldResult]

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
        self.parameter_grid = parameter_grid
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

        self.outer_splitter = outer_splitter_cls(
            n_splits=k_outer,
            shuffle=shuffle_outer,
            random_state=random_state,
        )

        self.inner_splitter = inner_splitter_cls(
            n_splits=k_inner,
            shuffle=shuffle_inner,
            random_state=random_state,
        )

        if dataloader_factory is None:
            self.dataloader_factory = lambda ds, shuffle: DataLoader(ds, batch_size=1, shuffle=shuffle)
        else:
            self.dataloader_factory = dataloader_factory

        self._validate_parameter_grid()

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
                continue

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
    ) -> TrialResult:
        params = self.suggest_parameters(trial, self.parameter_grid)
        _, _, trainer = self._build_trainer_for_trial(params=params)

        inner_results: list[InnerFoldResult] = []
        inner_metrics: list[float] = []
        inner_oof_logits_all: list[dict[str, torch.Tensor]] = []
        inner_oof_targets_all: list[dict[str, torch.Tensor]] = []

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            self.inner_splitter.split(outer_train_dataset, outer_train_index, outer_train_groups)
        ):
            train_subset = Subset(outer_train_dataset, inner_train_idx)
            val_subset = Subset(outer_train_dataset, inner_val_idx)

            train_loader = self.dataloader_factory(train_subset, True)
            val_loader = self.dataloader_factory(val_subset, False)

            trainer.reset_state()
            trainer.fit(
                train_loader,
                val_loader,
                trial=None,
            )

            metric = trainer.state.best_metric

            fold_result = InnerFoldResult(
                fold=inner_fold,
                best_metric=metric,
                best_epoch=trainer.state.best_epoch,
                best_state_dict_cpu=_clone_state_dict_cpu(trainer.state.best_state_dict_cpu),
                oof_logits=_clone_tensor_dict(trainer.state.oof_logits),
                oof_targets=_clone_tensor_dict(trainer.state.oof_targets),
            )
            inner_results.append(fold_result)

            if metric is not None:
                inner_metrics.append(float(metric))

            if trainer.state.oof_logits:
                inner_oof_logits_all.append(_clone_tensor_dict(trainer.state.oof_logits))
            if trainer.state.oof_targets:
                inner_oof_targets_all.append(_clone_tensor_dict(trainer.state.oof_targets))

        if len(inner_metrics) == 0:
            raise ValueError(f"Trial {trial.number} produced no valid inner-fold metrics.")

        aggregate_metric = sum(inner_metrics) / len(inner_metrics)
        aggregate_oof_logits = _concat_tensor_dicts(inner_oof_logits_all) if inner_oof_logits_all else {}
        aggregate_oof_targets = _concat_tensor_dicts(inner_oof_targets_all) if inner_oof_targets_all else {}

        return TrialResult(
            trial_number=trial.number,
            params=params,
            status="SUCCESS",
            aggregate_metric=float(aggregate_metric),
            inner_results=inner_results,
            aggregate_oof_logits=aggregate_oof_logits,
            aggregate_oof_targets=aggregate_oof_targets,
            error_message=None,
        )

    def run(
        self,
        dataset: TorchkitDataset,
        index: Any = None,
        groups: Optional[Any] = None,
    ) -> NestedCVResult:
        outer_results: list[OuterFoldResult] = []

        for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
            self.outer_splitter.split(dataset, index, groups)
        ):
            outer_train_dataset = Subset(dataset, outer_train_idx)
            outer_test_dataset = Subset(dataset, outer_test_idx)

            outer_train_index = _safe_take(index, outer_train_idx) if index is not None else None
            outer_train_groups = _safe_take(groups, outer_train_idx) if groups is not None else None

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
                    )
                    assert trial_result.aggregate_metric is not None
                    study.tell(trial, trial_result.aggregate_metric)
                    trial_results.append(trial_result)
                    successful_trials += 1

                except optuna.TrialPruned as e:
                    study.tell(trial, state=TrialState.PRUNED)
                    trial_results.append(
                        TrialResult(
                            trial_number=trial.number,
                            params={},
                            status="PRUNED",
                            aggregate_metric=None,
                            inner_results=[],
                            aggregate_oof_logits={},
                            aggregate_oof_targets={},
                            error_message=str(e),
                        )
                    )
                    pruned_trials += 1

                except Exception as e:
                    study.tell(trial, state=TrialState.FAIL)
                    trial_results.append(
                        TrialResult(
                            trial_number=trial.number,
                            params={},
                            status="FAILED",
                            aggregate_metric=None,
                            inner_results=[],
                            aggregate_oof_logits={},
                            aggregate_oof_targets={},
                            error_message=f"{type(e).__name__}: {e}",
                        )
                    )
                    failed_trials += 1

            successful_trial_results = [tr for tr in trial_results if tr.status == "SUCCESS"]
            if len(successful_trial_results) == 0:
                raise RuntimeError(f"Outer fold {outer_fold} produced no successful trials.")

            best_params = study.best_params
            best_metric = float(study.best_value)
            best_trial_number = study.best_trial.number

            try:
                best_trial_result = next(
                    tr for tr in successful_trial_results if tr.trial_number == best_trial_number
                )
            except StopIteration as e:
                raise RuntimeError(
                    f"Best Optuna trial {best_trial_number} was not found in stored successful trial_results."
                ) from e

            selected_inner_metrics = [
                float(r.best_metric) for r in best_trial_result.inner_results if r.best_metric is not None
            ]
            selected_inner_metric_mean = (
                float(statistics.mean(selected_inner_metrics)) if selected_inner_metrics else None
            )
            selected_inner_metric_std = (
                float(statistics.stdev(selected_inner_metrics)) if len(selected_inner_metrics) >= 2 else 0.0 if len(selected_inner_metrics) == 1 else None
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
                    best_params=best_params,
                    best_metric=best_metric,
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
                    final_best_epoch=None,
                    final_best_metric=None,
                    final_model_state_dict_cpu=final_model_state_dict_cpu if self.keep_final_model_state_dict_cpu else None,
                    final_model_state_dict_path=final_model_state_dict_path,
                    test_metrics=test_metrics,
                )
            )

        return NestedCVResult(outer_results=outer_results)
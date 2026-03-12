from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

import copy
import json

import pandas as pd
import torch

from torchkit.models.Model.factory import TorchkitModelFactory, TorchkitModelSpec
from torchkit.train.cv._base_cv import MetricDirection
from torchkit.train.cv._optuna_search_mixin import SuggestionType, TrialStatus
from torchkit.train.factory import TrainerFactory, TrainerSpec
from torchkit.train.trainer import Trainer


def _tensor_summary(x: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
    }


def _tensor_dict_summary(d: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {k: _tensor_summary(v) for k, v in d.items()}


def _is_json_primitive(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


def _to_jsonable(x: Any) -> Any:
    if _is_json_primitive(x):
        return x

    if isinstance(x, Path):
        return str(x)

    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    if isinstance(x, set):
        return sorted(_to_jsonable(v) for v in x)

    if torch.is_tensor(x):
        return _tensor_summary(x)

    if is_dataclass(x):
        return _to_jsonable(asdict(x))

    return repr(x)


def _flatten_dict(
    d: dict[str, Any],
    *,
    prefix: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, prefix=key, sep=sep))
        else:
            out[key] = v
    return out


@dataclass
class FoldResult:
    fold: int

    train_indices: list[int] = field(default_factory=list)
    val_indices: list[int] = field(default_factory=list)

    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    best_state_dict_cpu: Optional[dict[str, torch.Tensor]] = None

    oof_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    oof_targets: dict[str, torch.Tensor] = field(default_factory=dict)
    oof_sample_indices: list[int] = field(default_factory=list)

    def to_dict(self, *, include_tensors: bool = False) -> dict[str, Any]:
        out = {
            "fold": self.fold,
            "train_indices": copy.deepcopy(self.train_indices),
            "val_indices": copy.deepcopy(self.val_indices),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "oof_sample_indices": copy.deepcopy(self.oof_sample_indices),
            "n_train": len(self.train_indices),
            "n_val": len(self.val_indices),
        }

        if include_tensors:
            out["best_state_dict_cpu"] = {
                k: v.detach().cpu().clone() for k, v in (self.best_state_dict_cpu or {}).items()
            }
            out["oof_logits"] = {k: v.detach().cpu().clone() for k, v in self.oof_logits.items()}
            out["oof_targets"] = {k: v.detach().cpu().clone() for k, v in self.oof_targets.items()}
        else:
            out["best_state_dict_cpu"] = (
                None if self.best_state_dict_cpu is None
                else _tensor_dict_summary(self.best_state_dict_cpu)
            )
            out["oof_logits"] = _tensor_dict_summary(self.oof_logits)
            out["oof_targets"] = _tensor_dict_summary(self.oof_targets)

        return out

    def leaderboard_row(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "n_train": len(self.train_indices),
            "n_val": len(self.val_indices),
            "n_oof": len(self.oof_sample_indices),
        }


@dataclass
class OptunaTrialResult:
    trial_number: int
    params: dict[str, Any]
    status: TrialStatus

    aggregate_metric: Optional[float]
    aggregate_selection_score: Optional[float]

    fold_results: list[FoldResult] = field(default_factory=list)

    aggregate_oof_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    aggregate_oof_targets: dict[str, torch.Tensor] = field(default_factory=dict)
    aggregate_oof_sample_indices: list[int] = field(default_factory=list)

    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(
        self,
        *,
        include_tensors: bool = False,
        include_traceback: bool = False,
    ) -> dict[str, Any]:
        out = {
            "trial_number": self.trial_number,
            "params": copy.deepcopy(self.params),
            "status": self.status,
            "aggregate_metric": self.aggregate_metric,
            "aggregate_selection_score": self.aggregate_selection_score,
            "aggregate_oof_sample_indices": copy.deepcopy(self.aggregate_oof_sample_indices),
            "error_message": self.error_message,
            "fold_results": [fr.to_dict(include_tensors=include_tensors) for fr in self.fold_results],
        }

        if include_traceback:
            out["error_traceback"] = self.error_traceback

        if include_tensors:
            out["aggregate_oof_logits"] = {
                k: v.detach().cpu().clone() for k, v in self.aggregate_oof_logits.items()
            }
            out["aggregate_oof_targets"] = {
                k: v.detach().cpu().clone() for k, v in self.aggregate_oof_targets.items()
            }
        else:
            out["aggregate_oof_logits"] = _tensor_dict_summary(self.aggregate_oof_logits)
            out["aggregate_oof_targets"] = _tensor_dict_summary(self.aggregate_oof_targets)

        return out

    def leaderboard_row(self) -> dict[str, Any]:
        row = {
            "trial_number": self.trial_number,
            "status": self.status,
            "aggregate_metric": self.aggregate_metric,
            "aggregate_selection_score": self.aggregate_selection_score,
            "n_fold_results": len(self.fold_results),
            "n_aggregate_oof": len(self.aggregate_oof_sample_indices),
            "error_message": self.error_message,
        }
        row.update({f"param.{k}": v for k, v in self.params.items()})
        return row

    def folds_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([fr.leaderboard_row() for fr in self.fold_results])


@dataclass
class OptunaSearchCVResult:
    # Split membership for the search pool itself
    search_pool_indices: list[int] = field(default_factory=list)

    trial_results: list[OptunaTrialResult] = field(default_factory=list)

    best_params: dict[str, Any] = field(default_factory=dict)
    best_metric: float = 0.0
    best_selection_score: float = 0.0
    best_trial_number: int = -1

    attempted_trials: int = 0
    successful_trials: int = 0
    failed_trials: int = 0
    pruned_trials: int = 0

    selected_fold_results: list[FoldResult] = field(default_factory=list)
    selected_metric_mean: Optional[float] = None
    selected_metric_std: Optional[float] = None
    selected_metric_min: Optional[float] = None
    selected_metric_max: Optional[float] = None

    final_model_spec: Optional[TorchkitModelSpec] = None
    final_trainer_spec: Optional[TrainerSpec] = None

    final_fit_epochs: Optional[int] = None
    final_epochs_ran: Optional[int] = None
    final_best_epoch: Optional[int] = None
    final_best_metric: Optional[float] = None

    final_train_logs: list[dict[str, Any]] = field(default_factory=list)
    final_val_logs: list[dict[str, Any]] = field(default_factory=list)
    final_history: list[dict[str, Any]] = field(default_factory=list)

    final_model_state_dict_cpu: Optional[dict[str, torch.Tensor]] = None
    final_model_state_dict_path: Optional[str] = None

    holdout_metrics: Optional[dict[str, Any]] = None

    # CV-level metadata
    base_model_spec: Optional[TorchkitModelSpec] = None
    base_trainer_spec: Optional[TrainerSpec] = None
    parameter_grid: dict[str, tuple[list, SuggestionType]] = field(default_factory=dict)

    splitter_name: str = ""
    n_splits: int = 0
    shuffle: bool = False
    random_state: Optional[int] = None

    n_trials: int = 0
    max_trial_attempts: int = 0
    calibrate: bool = True
    final_model_dir: Optional[str] = None
    keep_final_model_state_dict_cpu: bool = True

    selection_metric_name: str = ""
    selection_metric_direction: MetricDirection = "maximize"

    def rebuild_final_model(
        self,
        *,
        device: torch.device | str = "cpu",
    ):
        if self.final_model_spec is None:
            raise ValueError("Result does not contain final_model_spec.")

        if self.final_model_state_dict_path is not None:
            return TorchkitModelFactory.build(
                copy.deepcopy(self.final_model_spec),
                state_dict_path=self.final_model_state_dict_path,
                device=device,
            )

        if self.final_model_state_dict_cpu is not None:
            return TorchkitModelFactory.build(
                copy.deepcopy(self.final_model_spec),
                state_dict=self.final_model_state_dict_cpu,
                device=device,
            )

        raise ValueError(
            "Result does not contain a saved final model state_dict "
            "(neither in-memory nor on disk)."
        )

    def rebuild_final_trainer(
        self,
        *,
        device: torch.device | str = "cpu",
    ) -> Trainer:
        if self.final_trainer_spec is None:
            raise ValueError("Result does not contain final_trainer_spec.")

        model = self.rebuild_final_model(device=device)

        trainer_spec = copy.deepcopy(self.final_trainer_spec)
        trainer_spec.config.device = device

        return TrainerFactory.build(
            trainer_spec,
            model=model,
        )

    def successful_trials(self) -> list[OptunaTrialResult]:
        return [tr for tr in self.trial_results if tr.status == "SUCCESS"]

    def selected_trial_result(self) -> OptunaTrialResult:
        try:
            return next(tr for tr in self.trial_results if tr.trial_number == self.best_trial_number)
        except StopIteration as e:
            raise ValueError(
                f"Best trial number {self.best_trial_number} not found in stored trial_results."
            ) from e

    def to_dict(
        self,
        *,
        include_tensors: bool = False,
        include_tracebacks: bool = False,
        include_specs_repr: bool = True,
    ) -> dict[str, Any]:
        out = {
            "search_pool_indices": copy.deepcopy(self.search_pool_indices),
            "trial_results": [
                tr.to_dict(
                    include_tensors=include_tensors,
                    include_traceback=include_tracebacks,
                )
                for tr in self.trial_results
            ],
            "best_params": copy.deepcopy(self.best_params),
            "best_metric": self.best_metric,
            "best_selection_score": self.best_selection_score,
            "best_trial_number": self.best_trial_number,
            "attempted_trials": self.attempted_trials,
            "successful_trials": self.successful_trials,
            "failed_trials": self.failed_trials,
            "pruned_trials": self.pruned_trials,
            "selected_fold_results": [
                fr.to_dict(include_tensors=include_tensors) for fr in self.selected_fold_results
            ],
            "selected_metric_mean": self.selected_metric_mean,
            "selected_metric_std": self.selected_metric_std,
            "selected_metric_min": self.selected_metric_min,
            "selected_metric_max": self.selected_metric_max,
            "final_fit_epochs": self.final_fit_epochs,
            "final_epochs_ran": self.final_epochs_ran,
            "final_best_epoch": self.final_best_epoch,
            "final_best_metric": self.final_best_metric,
            "final_train_logs": copy.deepcopy(self.final_train_logs),
            "final_val_logs": copy.deepcopy(self.final_val_logs),
            "final_history": copy.deepcopy(self.final_history),
            "final_model_state_dict_path": self.final_model_state_dict_path,
            "holdout_metrics": copy.deepcopy(self.holdout_metrics),
            "parameter_grid": copy.deepcopy(self.parameter_grid),
            "splitter_name": self.splitter_name,
            "n_splits": self.n_splits,
            "shuffle": self.shuffle,
            "random_state": self.random_state,
            "n_trials": self.n_trials,
            "max_trial_attempts": self.max_trial_attempts,
            "calibrate": self.calibrate,
            "final_model_dir": self.final_model_dir,
            "keep_final_model_state_dict_cpu": self.keep_final_model_state_dict_cpu,
            "selection_metric_name": self.selection_metric_name,
            "selection_metric_direction": self.selection_metric_direction,
        }

        if include_specs_repr:
            out["base_model_spec_repr"] = None if self.base_model_spec is None else repr(self.base_model_spec)
            out["base_trainer_spec_repr"] = None if self.base_trainer_spec is None else repr(self.base_trainer_spec)
            out["final_model_spec_repr"] = None if self.final_model_spec is None else repr(self.final_model_spec)
            out["final_trainer_spec_repr"] = None if self.final_trainer_spec is None else repr(self.final_trainer_spec)

        if include_tensors:
            out["final_model_state_dict_cpu"] = (
                None
                if self.final_model_state_dict_cpu is None
                else {k: v.detach().cpu().clone() for k, v in self.final_model_state_dict_cpu.items()}
            )
        else:
            out["final_model_state_dict_cpu"] = (
                None
                if self.final_model_state_dict_cpu is None
                else _tensor_dict_summary(self.final_model_state_dict_cpu)
            )

        return out

    def to_json(
        self,
        path: str | Path | None = None,
        *,
        indent: int = 2,
        include_tensors: bool = False,
        include_tracebacks: bool = False,
        include_specs_repr: bool = True,
    ) -> str:
        payload = self.to_dict(
            include_tensors=include_tensors,
            include_tracebacks=include_tracebacks,
            include_specs_repr=include_specs_repr,
        )
        text = json.dumps(_to_jsonable(payload), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def trials_to_dataframe(self) -> pd.DataFrame:
        rows = [tr.leaderboard_row() for tr in self.trial_results]
        return pd.DataFrame(rows)

    def folds_to_dataframe(self, *, selected_only: bool = True) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        if selected_only:
            for fr in self.selected_fold_results:
                row = fr.leaderboard_row()
                row["trial_number"] = self.best_trial_number
                row["is_selected_trial"] = True
                rows.append(row)
        else:
            for tr in self.trial_results:
                for fr in tr.fold_results:
                    row = fr.leaderboard_row()
                    row["trial_number"] = tr.trial_number
                    row["trial_status"] = tr.status
                    row["is_selected_trial"] = tr.trial_number == self.best_trial_number
                    rows.append(row)

        return pd.DataFrame(rows)

    def leaderboard_row(self) -> dict[str, Any]:
        row = {
            "splitter_name": self.splitter_name,
            "n_splits": self.n_splits,
            "selection_metric_name": self.selection_metric_name,
            "selection_metric_direction": self.selection_metric_direction,
            "best_trial_number": self.best_trial_number,
            "best_metric": self.best_metric,
            "best_selection_score": self.best_selection_score,
            "selected_metric_mean": self.selected_metric_mean,
            "selected_metric_std": self.selected_metric_std,
            "selected_metric_min": self.selected_metric_min,
            "selected_metric_max": self.selected_metric_max,
            "attempted_trials": self.attempted_trials,
            "successful_trials": self.successful_trials,
            "failed_trials": self.failed_trials,
            "pruned_trials": self.pruned_trials,
            "final_fit_epochs": self.final_fit_epochs,
            "final_epochs_ran": self.final_epochs_ran,
        }

        if self.holdout_metrics is not None:
            row.update({f"holdout.{k}": v for k, v in self.holdout_metrics.items()})

        row.update({f"best_param.{k}": v for k, v in self.best_params.items()})
        return row


@dataclass
class OuterFoldResult:
    fold: int

    outer_train_indices: list[int] = field(default_factory=list)
    outer_test_indices: list[int] = field(default_factory=list)

    inner_search_result: OptunaSearchCVResult = field(default_factory=OptunaSearchCVResult)
    outer_test_metrics: Optional[dict[str, Any]] = None

    @property
    def best_params(self) -> dict[str, Any]:
        return self.inner_search_result.best_params

    @property
    def best_metric(self) -> float:
        return self.inner_search_result.best_metric

    @property
    def best_selection_score(self) -> float:
        return self.inner_search_result.best_selection_score

    @property
    def best_trial_number(self) -> int:
        return self.inner_search_result.best_trial_number

    def to_dict(
        self,
        *,
        include_tensors: bool = False,
        include_tracebacks: bool = False,
        include_inner_search_result: bool = True,
    ) -> dict[str, Any]:
        out = {
            "fold": self.fold,
            "outer_train_indices": copy.deepcopy(self.outer_train_indices),
            "outer_test_indices": copy.deepcopy(self.outer_test_indices),
            "outer_test_metrics": copy.deepcopy(self.outer_test_metrics),
            "n_outer_train": len(self.outer_train_indices),
            "n_outer_test": len(self.outer_test_indices),
        }

        if include_inner_search_result:
            out["inner_search_result"] = self.inner_search_result.to_dict(
                include_tensors=include_tensors,
                include_tracebacks=include_tracebacks,
            )
        else:
            out["inner_search_summary"] = self.inner_search_result.leaderboard_row()

        return out

    def leaderboard_row(self) -> dict[str, Any]:
        row = {
            "outer_fold": self.fold,
            "n_outer_train": len(self.outer_train_indices),
            "n_outer_test": len(self.outer_test_indices),
        }
        if self.outer_test_metrics is not None:
            row.update({f"outer_test.{k}": v for k, v in self.outer_test_metrics.items()})

        row.update({f"inner.{k}": v for k, v in self.inner_search_result.leaderboard_row().items()})
        return row


@dataclass
class NestedOptunaSearchCVResult:
    outer_results: list[OuterFoldResult] = field(default_factory=list)

    # CV-level metadata for offline reporting / auditability
    base_model_spec: Optional[TorchkitModelSpec] = None
    base_trainer_spec: Optional[TrainerSpec] = None
    parameter_grid: dict[str, tuple[list, SuggestionType]] = field(default_factory=dict)

    outer_splitter_name: str = ""
    inner_splitter_name: str = ""
    k_outer: int = 0
    k_inner: int = 0
    shuffle_outer: bool = False
    shuffle_inner: bool = False
    random_state: Optional[int] = None

    n_trials: int = 0
    max_trial_attempts: int = 0
    calibrate: bool = True
    final_model_dir: Optional[str] = None
    keep_final_model_state_dict_cpu: bool = True

    selection_metric_name: str = ""
    selection_metric_direction: MetricDirection = "maximize"

    def rebuild_final_model(
        self,
        outer_fold: int,
        *,
        device: torch.device | str = "cpu",
    ):
        return self.outer_results[outer_fold].inner_search_result.rebuild_final_model(device=device)

    def rebuild_final_trainer(
        self,
        outer_fold: int,
        *,
        device: torch.device | str = "cpu",
    ) -> Trainer:
        return self.outer_results[outer_fold].inner_search_result.rebuild_final_trainer(device=device)

    def to_dict(
        self,
        *,
        include_tensors: bool = False,
        include_tracebacks: bool = False,
        include_inner_search_results: bool = True,
        include_specs_repr: bool = True,
    ) -> dict[str, Any]:
        out = {
            "outer_results": [
                r.to_dict(
                    include_tensors=include_tensors,
                    include_tracebacks=include_tracebacks,
                    include_inner_search_result=include_inner_search_results,
                )
                for r in self.outer_results
            ],
            "parameter_grid": copy.deepcopy(self.parameter_grid),
            "outer_splitter_name": self.outer_splitter_name,
            "inner_splitter_name": self.inner_splitter_name,
            "k_outer": self.k_outer,
            "k_inner": self.k_inner,
            "shuffle_outer": self.shuffle_outer,
            "shuffle_inner": self.shuffle_inner,
            "random_state": self.random_state,
            "n_trials": self.n_trials,
            "max_trial_attempts": self.max_trial_attempts,
            "calibrate": self.calibrate,
            "final_model_dir": self.final_model_dir,
            "keep_final_model_state_dict_cpu": self.keep_final_model_state_dict_cpu,
            "selection_metric_name": self.selection_metric_name,
            "selection_metric_direction": self.selection_metric_direction,
        }

        if include_specs_repr:
            out["base_model_spec_repr"] = None if self.base_model_spec is None else repr(self.base_model_spec)
            out["base_trainer_spec_repr"] = None if self.base_trainer_spec is None else repr(self.base_trainer_spec)

        return out

    def to_json(
        self,
        path: str | Path | None = None,
        *,
        indent: int = 2,
        include_tensors: bool = False,
        include_tracebacks: bool = False,
        include_inner_search_results: bool = True,
        include_specs_repr: bool = True,
    ) -> str:
        payload = self.to_dict(
            include_tensors=include_tensors,
            include_tracebacks=include_tracebacks,
            include_inner_search_results=include_inner_search_results,
            include_specs_repr=include_specs_repr,
        )
        text = json.dumps(_to_jsonable(payload), indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def outer_folds_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.leaderboard_row() for r in self.outer_results])

    def leaderboard_row(self) -> dict[str, Any]:
        row = {
            "outer_splitter_name": self.outer_splitter_name,
            "inner_splitter_name": self.inner_splitter_name,
            "k_outer": self.k_outer,
            "k_inner": self.k_inner,
            "selection_metric_name": self.selection_metric_name,
            "selection_metric_direction": self.selection_metric_direction,
            "n_trials": self.n_trials,
            "max_trial_attempts": self.max_trial_attempts,
        }

        if not self.outer_results:
            return row

        # Aggregate numeric outer test metrics where possible
        metric_buckets: dict[str, list[float]] = {}
        for outer in self.outer_results:
            if outer.outer_test_metrics is None:
                continue
            for k, v in outer.outer_test_metrics.items():
                if isinstance(v, (int, float)):
                    metric_buckets.setdefault(k, []).append(float(v))

        for k, vals in metric_buckets.items():
            if vals:
                row[f"outer_test.{k}.mean"] = sum(vals) / len(vals)
                row[f"outer_test.{k}.min"] = min(vals)
                row[f"outer_test.{k}.max"] = max(vals)
                if len(vals) >= 2:
                    mean = sum(vals) / len(vals)
                    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
                    row[f"outer_test.{k}.std"] = var ** 0.5
                else:
                    row[f"outer_test.{k}.std"] = 0.0

        return row
from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import copy
import json
import os

import pandas as pd
import torch

from torchkit.models.Model.factory import TorchkitModelFactory, TorchkitModelSpec
from torchkit.evaluate.report.bundle import BundleReportEvaluator
from torchkit.train.cv._base_cv import MetricDirection
from torchkit.train.cv._optuna_search_mixin import ParameterGrid, TrialStatus
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


def _clone_tensor_dict_for_storage(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in d.items()}


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
        return _to_jsonable(_snapshot_object(x))

    return repr(x)


def _flatten_metric_mapping(
    value: Optional[dict[str, Any]],
    *,
    prefix: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    return {f"{prefix}.{k}": v for k, v in _flatten_dict(value).items()}


def _qualified_name(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else obj.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _snapshot_object(x: Any) -> Any:
    if _is_json_primitive(x):
        return x

    if isinstance(x, Path):
        return {"__kind__": "path", "value": str(x)}

    if isinstance(x, dict):
        return {str(k): _snapshot_object(v) for k, v in x.items()}

    if isinstance(x, list):
        return [_snapshot_object(v) for v in x]

    if isinstance(x, tuple):
        return {
            "__kind__": "tuple",
            "items": [_snapshot_object(v) for v in x],
        }

    if isinstance(x, set):
        return {
            "__kind__": "set",
            "items": sorted(_snapshot_object(v) for v in x),
        }

    if torch.is_tensor(x):
        return {
            "__kind__": "tensor_summary",
            **_tensor_summary(x),
        }

    if isinstance(x, type):
        return {
            "__kind__": "type",
            "qualified_name": _qualified_name(x),
            "repr": repr(x),
        }

    if callable(x) and hasattr(x, "__module__") and hasattr(x, "__qualname__"):
        return {
            "__kind__": "callable",
            "qualified_name": f"{x.__module__}.{x.__qualname__}",
            "repr": repr(x),
        }

    if is_dataclass(x):
        return {
            "__kind__": "dataclass",
            "qualified_name": _qualified_name(x),
            "fields": {
                f.name: _snapshot_object(getattr(x, f.name))
                for f in dataclass_fields(x)
            },
        }

    state = None
    if hasattr(x, "__dict__"):
        state = {
            str(k): _snapshot_object(v)
            for k, v in vars(x).items()
        }

    out = {
        "__kind__": "object",
        "qualified_name": _qualified_name(x),
        "repr": repr(x),
    }
    if state is not None:
        out["state"] = state
    return out


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
    report_results: Optional[dict[str, Any]] = None
    log_file: Optional[str] = None
    tensor_artifact_path: Optional[str] = None
    best_state_dict_cpu_summary: Optional[dict[str, Any]] = None
    oof_logits_summary: dict[str, Any] = field(default_factory=dict)
    oof_targets_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._refresh_tensor_summaries()

    def _refresh_tensor_summaries(self) -> None:
        if self.best_state_dict_cpu is not None:
            self.best_state_dict_cpu_summary = _tensor_dict_summary(self.best_state_dict_cpu)
        elif self.best_state_dict_cpu_summary is None:
            self.best_state_dict_cpu_summary = None

        if self.oof_logits:
            self.oof_logits_summary = _tensor_dict_summary(self.oof_logits)
        elif not self.oof_logits_summary:
            self.oof_logits_summary = {}

        if self.oof_targets:
            self.oof_targets_summary = _tensor_dict_summary(self.oof_targets)
        elif not self.oof_targets_summary:
            self.oof_targets_summary = {}

    def has_in_memory_tensors(self) -> bool:
        return (
            self.best_state_dict_cpu is not None
            or bool(self.oof_logits)
            or bool(self.oof_targets)
        )

    def hydrate_tensors(self) -> None:
        if self.has_in_memory_tensors() or self.tensor_artifact_path is None:
            self._refresh_tensor_summaries()
            return

        payload = torch.load(self.tensor_artifact_path, map_location="cpu")
        best_state_dict_cpu = payload.get("best_state_dict_cpu")
        self.best_state_dict_cpu = (
            None if best_state_dict_cpu is None else _clone_tensor_dict_for_storage(best_state_dict_cpu)
        )
        self.oof_logits = _clone_tensor_dict_for_storage(dict(payload.get("oof_logits", {})))
        self.oof_targets = _clone_tensor_dict_for_storage(dict(payload.get("oof_targets", {})))
        self._refresh_tensor_summaries()

    def spill_tensors(self, path: str, *, release_memory: bool = True) -> None:
        self._refresh_tensor_summaries()
        payload = {
            "best_state_dict_cpu": (
                None
                if self.best_state_dict_cpu is None
                else _clone_tensor_dict_for_storage(self.best_state_dict_cpu)
            ),
            "oof_logits": _clone_tensor_dict_for_storage(self.oof_logits),
            "oof_targets": _clone_tensor_dict_for_storage(self.oof_targets),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        self.tensor_artifact_path = path
        if release_memory:
            self.best_state_dict_cpu = None
            self.oof_logits = {}
            self.oof_targets = {}

    def to_dict(self, *, include_tensors: bool = False) -> dict[str, Any]:
        if include_tensors:
            self.hydrate_tensors()
        else:
            self._refresh_tensor_summaries()

        out = {
            "fold": self.fold,
            "train_indices": copy.deepcopy(self.train_indices),
            "val_indices": copy.deepcopy(self.val_indices),
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "oof_sample_indices": copy.deepcopy(self.oof_sample_indices),
            "report_results": copy.deepcopy(self.report_results),
            "log_file": self.log_file,
            "n_train": len(self.train_indices),
            "n_val": len(self.val_indices),
            "tensor_artifact_path": self.tensor_artifact_path,
        }

        if include_tensors:
            out["best_state_dict_cpu"] = {
                k: v.detach().cpu().clone() for k, v in (self.best_state_dict_cpu or {}).items()
            }
            out["oof_logits"] = {k: v.detach().cpu().clone() for k, v in self.oof_logits.items()}
            out["oof_targets"] = {k: v.detach().cpu().clone() for k, v in self.oof_targets.items()}
        else:
            out["best_state_dict_cpu"] = copy.deepcopy(self.best_state_dict_cpu_summary)
            out["oof_logits"] = copy.deepcopy(self.oof_logits_summary)
            out["oof_targets"] = copy.deepcopy(self.oof_targets_summary)

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FoldResult":
        return cls(
            fold=int(payload.get("fold", 0)),
            train_indices=list(payload.get("train_indices", [])),
            val_indices=list(payload.get("val_indices", [])),
            best_metric=payload.get("best_metric"),
            best_epoch=payload.get("best_epoch"),
            best_state_dict_cpu=None,
            oof_logits={},
            oof_targets={},
            oof_sample_indices=list(payload.get("oof_sample_indices", [])),
            report_results=copy.deepcopy(payload.get("report_results")),
            log_file=payload.get("log_file"),
            tensor_artifact_path=payload.get("tensor_artifact_path"),
            best_state_dict_cpu_summary=copy.deepcopy(payload.get("best_state_dict_cpu")),
            oof_logits_summary=copy.deepcopy(payload.get("oof_logits", {})),
            oof_targets_summary=copy.deepcopy(payload.get("oof_targets", {})),
        )


@dataclass
class OptunaTrialResult:
    trial_number: int
    params: dict[str, Any]
    status: TrialStatus

    aggregate_metric: Optional[float]
    aggregate_selection_score: Optional[float]
    intermediate_reports: list[dict[str, Any]] = field(default_factory=list)
    pruned_epoch: Optional[int] = None

    fold_results: list[FoldResult] = field(default_factory=list)
    aggregate_fold_report_results: Optional[dict[str, list[Any]]] = None
    log_file: Optional[str] = None

    aggregate_oof_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    aggregate_oof_targets: dict[str, torch.Tensor] = field(default_factory=dict)
    aggregate_oof_sample_indices: list[int] = field(default_factory=list)

    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    aggregate_tensor_artifact_path: Optional[str] = None
    aggregate_oof_logits_summary: dict[str, Any] = field(default_factory=dict)
    aggregate_oof_targets_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._refresh_tensor_summaries()

    def _refresh_tensor_summaries(self) -> None:
        if self.aggregate_oof_logits:
            self.aggregate_oof_logits_summary = _tensor_dict_summary(self.aggregate_oof_logits)
        elif not self.aggregate_oof_logits_summary:
            self.aggregate_oof_logits_summary = {}

        if self.aggregate_oof_targets:
            self.aggregate_oof_targets_summary = _tensor_dict_summary(self.aggregate_oof_targets)
        elif not self.aggregate_oof_targets_summary:
            self.aggregate_oof_targets_summary = {}

    def hydrate_tensors(self) -> None:
        for fold_result in self.fold_results:
            fold_result.hydrate_tensors()

        if self.aggregate_oof_logits or self.aggregate_oof_targets or self.aggregate_tensor_artifact_path is None:
            self._refresh_tensor_summaries()
            return

        payload = torch.load(self.aggregate_tensor_artifact_path, map_location="cpu")
        self.aggregate_oof_logits = _clone_tensor_dict_for_storage(dict(payload.get("aggregate_oof_logits", {})))
        self.aggregate_oof_targets = _clone_tensor_dict_for_storage(dict(payload.get("aggregate_oof_targets", {})))
        self._refresh_tensor_summaries()

    def spill_tensors(self, artifact_dir: str, *, release_memory: bool = True) -> None:
        Path(artifact_dir).mkdir(parents=True, exist_ok=True)
        for fold_result in self.fold_results:
            fold_result.spill_tensors(
                os.path.join(artifact_dir, f"fold_{fold_result.fold:03d}_tensors.pt"),
                release_memory=release_memory,
            )

        self._refresh_tensor_summaries()
        aggregate_path = os.path.join(artifact_dir, "aggregate_oof_tensors.pt")
        payload = {
            "aggregate_oof_logits": _clone_tensor_dict_for_storage(self.aggregate_oof_logits),
            "aggregate_oof_targets": _clone_tensor_dict_for_storage(self.aggregate_oof_targets),
        }
        torch.save(payload, aggregate_path)
        self.aggregate_tensor_artifact_path = aggregate_path
        if release_memory:
            self.aggregate_oof_logits = {}
            self.aggregate_oof_targets = {}

    def to_dict(
        self,
        *,
        include_tensors: bool = False,
        include_traceback: bool = False,
    ) -> dict[str, Any]:
        if include_tensors:
            self.hydrate_tensors()
        else:
            self._refresh_tensor_summaries()

        out = {
            "trial_number": self.trial_number,
            "params": copy.deepcopy(self.params),
            "status": self.status,
            "aggregate_metric": self.aggregate_metric,
            "aggregate_selection_score": self.aggregate_selection_score,
            "intermediate_reports": copy.deepcopy(self.intermediate_reports),
            "pruned_epoch": self.pruned_epoch,
            "aggregate_fold_report_results": copy.deepcopy(self.aggregate_fold_report_results),
            "log_file": self.log_file,
            "aggregate_oof_sample_indices": copy.deepcopy(self.aggregate_oof_sample_indices),
            "error_message": self.error_message,
            "aggregate_tensor_artifact_path": self.aggregate_tensor_artifact_path,
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
            out["aggregate_oof_logits"] = copy.deepcopy(self.aggregate_oof_logits_summary)
            out["aggregate_oof_targets"] = copy.deepcopy(self.aggregate_oof_targets_summary)

        return out

    def leaderboard_row(self) -> dict[str, Any]:
        row = {
            "trial_number": self.trial_number,
            "status": self.status,
            "aggregate_metric": self.aggregate_metric,
            "aggregate_selection_score": self.aggregate_selection_score,
            "n_intermediate_reports": len(self.intermediate_reports),
            "pruned_epoch": self.pruned_epoch,
            "n_fold_results": len(self.fold_results),
            "n_aggregate_oof": len(self.aggregate_oof_sample_indices),
            "error_message": self.error_message,
        }
        row.update({f"param.{k}": v for k, v in self.params.items()})
        return row

    def folds_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([fr.leaderboard_row() for fr in self.fold_results])

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OptunaTrialResult":
        return cls(
            trial_number=int(payload.get("trial_number", -1)),
            params=copy.deepcopy(payload.get("params", {})),
            status=payload.get("status", "FAILED"),
            aggregate_metric=payload.get("aggregate_metric"),
            aggregate_selection_score=payload.get("aggregate_selection_score"),
            intermediate_reports=copy.deepcopy(payload.get("intermediate_reports", [])),
            pruned_epoch=payload.get("pruned_epoch"),
            fold_results=[
                FoldResult.from_dict(fr_payload)
                for fr_payload in payload.get("fold_results", [])
            ],
            aggregate_fold_report_results=copy.deepcopy(payload.get("aggregate_fold_report_results")),
            log_file=payload.get("log_file"),
            aggregate_oof_sample_indices=list(payload.get("aggregate_oof_sample_indices", [])),
            error_message=payload.get("error_message"),
            error_traceback=payload.get("error_traceback"),
            aggregate_tensor_artifact_path=payload.get("aggregate_tensor_artifact_path"),
            aggregate_oof_logits_summary=copy.deepcopy(payload.get("aggregate_oof_logits", {})),
            aggregate_oof_targets_summary=copy.deepcopy(payload.get("aggregate_oof_targets", {})),
        )


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
    selected_fold_report_results: Optional[dict[str, list[Any]]] = None
    selected_fold_report_results_raw: Optional[dict[str, list[Any]]] = None
    selected_metric_mean: Optional[float] = None
    selected_metric_std: Optional[float] = None
    selected_metric_min: Optional[float] = None
    selected_metric_max: Optional[float] = None
    selection_diagnostics: Optional[dict[str, Any]] = None
    selected_competence_summary: Optional[dict[str, Any]] = None
    fold_best_epochs: Optional[list[int]] = None

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
    holdout_metrics_by_phase: Optional[dict[str, dict[str, Any]]] = None
    holdout_report_results: Optional[dict[str, Any]] = None
    holdout_report_results_by_phase: Optional[dict[str, dict[str, Any]]] = None
    holdout_posthoc_results: Optional[dict[str, Any]] = None
    final_posthoc_module_summary: Optional[dict[str, Any]] = None

    # CV-level metadata
    base_model_spec: Optional[TorchkitModelSpec] = None
    base_trainer_spec: Optional[TrainerSpec] = None
    parameter_grid: Optional[ParameterGrid] = None
    report_evaluator: Optional[BundleReportEvaluator] = None
    posthoc_hooks: Optional[list[Any]] = None
    log_dir: Optional[str] = None
    run_log_file: Optional[str] = None
    final_refit_log_file: Optional[str] = None

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
    selection_metric_spec: dict[str, Any] = field(default_factory=dict)

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

    def successful_trial_results(self) -> list[OptunaTrialResult]:
        return [tr for tr in self.trial_results if tr.status == "SUCCESS"]

    def selected_trial_result(self) -> OptunaTrialResult:
        try:
            selected = next(tr for tr in self.trial_results if tr.trial_number == self.best_trial_number)
        except StopIteration as e:
            raise ValueError(
                f"Best trial number {self.best_trial_number} not found in stored trial_results."
            ) from e
        selected.hydrate_tensors()
        return selected

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
            "selected_fold_report_results": copy.deepcopy(self.selected_fold_report_results),
            "selected_fold_report_results_raw": copy.deepcopy(self.selected_fold_report_results_raw),
            "selected_metric_mean": self.selected_metric_mean,
            "selected_metric_std": self.selected_metric_std,
            "selected_metric_min": self.selected_metric_min,
            "selected_metric_max": self.selected_metric_max,
            "selection_diagnostics": copy.deepcopy(self.selection_diagnostics),
            "selected_competence_summary": copy.deepcopy(self.selected_competence_summary),
            "fold_best_epochs": copy.deepcopy(self.fold_best_epochs),
            "final_fit_epochs": self.final_fit_epochs,
            "final_epochs_ran": self.final_epochs_ran,
            "final_best_epoch": self.final_best_epoch,
            "final_best_metric": self.final_best_metric,
            "final_train_logs": copy.deepcopy(self.final_train_logs),
            "final_val_logs": copy.deepcopy(self.final_val_logs),
            "final_history": copy.deepcopy(self.final_history),
            "final_model_state_dict_path": self.final_model_state_dict_path,
            "holdout_metrics": copy.deepcopy(self.holdout_metrics),
            "holdout_metrics_by_phase": copy.deepcopy(self.holdout_metrics_by_phase),
            "holdout_report_results": copy.deepcopy(self.holdout_report_results),
            "holdout_report_results_by_phase": copy.deepcopy(self.holdout_report_results_by_phase),
            "holdout_posthoc_results": copy.deepcopy(self.holdout_posthoc_results),
            "final_posthoc_module_summary": copy.deepcopy(self.final_posthoc_module_summary),
            "parameter_grid": copy.deepcopy(self.parameter_grid),
            "report_evaluator": None if self.report_evaluator is None else _snapshot_object(self.report_evaluator),
            "posthoc_hooks": copy.deepcopy(self.posthoc_hooks),
            "log_dir": self.log_dir,
            "run_log_file": self.run_log_file,
            "final_refit_log_file": self.final_refit_log_file,
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
            "selection_metric_spec": copy.deepcopy(self.selection_metric_spec),
        }

        if include_specs_repr:
            out["base_model_spec_repr"] = None if self.base_model_spec is None else repr(self.base_model_spec)
            out["base_trainer_spec_repr"] = None if self.base_trainer_spec is None else repr(self.base_trainer_spec)
            out["final_model_spec_repr"] = None if self.final_model_spec is None else repr(self.final_model_spec)
            out["final_trainer_spec_repr"] = None if self.final_trainer_spec is None else repr(self.final_trainer_spec)

        out["base_model_spec"] = None if self.base_model_spec is None else _snapshot_object(self.base_model_spec)
        out["base_trainer_spec"] = None if self.base_trainer_spec is None else _snapshot_object(self.base_trainer_spec)
        out["final_model_spec"] = None if self.final_model_spec is None else _snapshot_object(self.final_model_spec)
        out["final_trainer_spec"] = None if self.final_trainer_spec is None else _snapshot_object(self.final_trainer_spec)

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
            row.update(_flatten_metric_mapping(self.holdout_metrics, prefix="holdout"))
        if self.holdout_posthoc_results is not None:
            row.update(_flatten_metric_mapping(self.holdout_posthoc_results, prefix="holdout_posthoc"))

        row.update({f"best_param.{k}": v for k, v in self.best_params.items()})
        return row

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OptunaSearchCVResult":
        return cls(
            search_pool_indices=list(payload.get("search_pool_indices", [])),
            trial_results=[
                OptunaTrialResult.from_dict(tr_payload)
                for tr_payload in payload.get("trial_results", [])
            ],
            best_params=copy.deepcopy(payload.get("best_params", {})),
            best_metric=payload.get("best_metric", 0.0),
            best_selection_score=payload.get("best_selection_score", 0.0),
            best_trial_number=int(payload.get("best_trial_number", -1)),
            attempted_trials=int(payload.get("attempted_trials", 0)),
            successful_trials=int(payload.get("successful_trials", 0)),
            failed_trials=int(payload.get("failed_trials", 0)),
            pruned_trials=int(payload.get("pruned_trials", 0)),
            selected_fold_results=[
                FoldResult.from_dict(fr_payload)
                for fr_payload in payload.get("selected_fold_results", [])
            ],
            selected_fold_report_results=copy.deepcopy(payload.get("selected_fold_report_results")),
            selected_fold_report_results_raw=copy.deepcopy(payload.get("selected_fold_report_results_raw")),
            selected_metric_mean=payload.get("selected_metric_mean"),
            selected_metric_std=payload.get("selected_metric_std"),
            selected_metric_min=payload.get("selected_metric_min"),
            selected_metric_max=payload.get("selected_metric_max"),
            selection_diagnostics=copy.deepcopy(payload.get("selection_diagnostics")),
            selected_competence_summary=copy.deepcopy(payload.get("selected_competence_summary")),
            fold_best_epochs=copy.deepcopy(payload.get("fold_best_epochs")),
            final_model_spec=copy.deepcopy(payload.get("final_model_spec")),
            final_trainer_spec=copy.deepcopy(payload.get("final_trainer_spec")),
            final_fit_epochs=payload.get("final_fit_epochs"),
            final_epochs_ran=payload.get("final_epochs_ran"),
            final_best_epoch=payload.get("final_best_epoch"),
            final_best_metric=payload.get("final_best_metric"),
            final_train_logs=copy.deepcopy(payload.get("final_train_logs", [])),
            final_val_logs=copy.deepcopy(payload.get("final_val_logs", [])),
            final_history=copy.deepcopy(payload.get("final_history", [])),
            final_model_state_dict_cpu=None,
            final_model_state_dict_path=payload.get("final_model_state_dict_path"),
            holdout_metrics=copy.deepcopy(payload.get("holdout_metrics")),
            holdout_metrics_by_phase=copy.deepcopy(payload.get("holdout_metrics_by_phase")),
            holdout_report_results=copy.deepcopy(payload.get("holdout_report_results")),
            holdout_report_results_by_phase=copy.deepcopy(payload.get("holdout_report_results_by_phase")),
            holdout_posthoc_results=copy.deepcopy(payload.get("holdout_posthoc_results")),
            final_posthoc_module_summary=copy.deepcopy(payload.get("final_posthoc_module_summary")),
            base_model_spec=copy.deepcopy(payload.get("base_model_spec")),
            base_trainer_spec=copy.deepcopy(payload.get("base_trainer_spec")),
            parameter_grid=copy.deepcopy(payload.get("parameter_grid")),
            report_evaluator=copy.deepcopy(payload.get("report_evaluator")),
            posthoc_hooks=copy.deepcopy(payload.get("posthoc_hooks")),
            log_dir=payload.get("log_dir"),
            run_log_file=payload.get("run_log_file"),
            final_refit_log_file=payload.get("final_refit_log_file"),
            splitter_name=payload.get("splitter_name", ""),
            n_splits=int(payload.get("n_splits", 0)),
            shuffle=bool(payload.get("shuffle", False)),
            random_state=payload.get("random_state"),
            n_trials=int(payload.get("n_trials", 0)),
            max_trial_attempts=int(payload.get("max_trial_attempts", 0)),
            calibrate=bool(payload.get("calibrate", True)),
            final_model_dir=payload.get("final_model_dir"),
            keep_final_model_state_dict_cpu=bool(payload.get("keep_final_model_state_dict_cpu", True)),
            selection_metric_name=payload.get("selection_metric_name", ""),
            selection_metric_direction=payload.get("selection_metric_direction", "maximize"),
            selection_metric_spec=copy.deepcopy(payload.get("selection_metric_spec", {})),
        )


@dataclass
class OuterFoldResult:
    fold: int

    outer_train_indices: list[int] = field(default_factory=list)
    outer_test_indices: list[int] = field(default_factory=list)

    inner_search_result: OptunaSearchCVResult = field(default_factory=OptunaSearchCVResult)
    outer_test_metrics: Optional[dict[str, Any]] = None
    outer_test_metrics_by_phase: Optional[dict[str, dict[str, Any]]] = None
    outer_test_report_results: Optional[dict[str, Any]] = None
    outer_test_report_results_by_phase: Optional[dict[str, dict[str, Any]]] = None
    outer_test_posthoc_results: Optional[dict[str, Any]] = None
    outer_test_posthoc_module_summary: Optional[dict[str, Any]] = None
    log_file: Optional[str] = None

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
            "outer_test_metrics_by_phase": copy.deepcopy(self.outer_test_metrics_by_phase),
            "outer_test_report_results": copy.deepcopy(self.outer_test_report_results),
            "outer_test_report_results_by_phase": copy.deepcopy(self.outer_test_report_results_by_phase),
            "outer_test_posthoc_results": copy.deepcopy(self.outer_test_posthoc_results),
            "outer_test_posthoc_module_summary": copy.deepcopy(self.outer_test_posthoc_module_summary),
            "log_file": self.log_file,
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
            row.update(_flatten_metric_mapping(self.outer_test_metrics, prefix="outer_test"))
        if self.outer_test_posthoc_results is not None:
            row.update(_flatten_metric_mapping(self.outer_test_posthoc_results, prefix="outer_test_posthoc"))

        row.update({f"inner.{k}": v for k, v in self.inner_search_result.leaderboard_row().items()})
        return row

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OuterFoldResult":
        inner_payload = payload.get("inner_search_result")
        if inner_payload is None:
            raise ValueError("OuterFoldResult.from_dict requires an 'inner_search_result' payload.")

        return cls(
            fold=int(payload.get("fold", 0)),
            outer_train_indices=list(payload.get("outer_train_indices", [])),
            outer_test_indices=list(payload.get("outer_test_indices", [])),
            inner_search_result=OptunaSearchCVResult.from_dict(inner_payload),
            outer_test_metrics=copy.deepcopy(payload.get("outer_test_metrics")),
            outer_test_metrics_by_phase=copy.deepcopy(payload.get("outer_test_metrics_by_phase")),
            outer_test_report_results=copy.deepcopy(payload.get("outer_test_report_results")),
            outer_test_report_results_by_phase=copy.deepcopy(payload.get("outer_test_report_results_by_phase")),
            outer_test_posthoc_results=copy.deepcopy(payload.get("outer_test_posthoc_results")),
            outer_test_posthoc_module_summary=copy.deepcopy(payload.get("outer_test_posthoc_module_summary")),
            log_file=payload.get("log_file"),
        )


@dataclass
class NestedOptunaSearchCVResult:
    outer_results: list[OuterFoldResult] = field(default_factory=list)

    # CV-level metadata for offline reporting / auditability
    base_model_spec: Optional[TorchkitModelSpec] = None
    base_trainer_spec: Optional[TrainerSpec] = None
    parameter_grid: Optional[ParameterGrid] = None
    report_evaluator: Optional[BundleReportEvaluator] = None
    outer_report_results: Optional[dict[str, list[Any]]] = None
    outer_posthoc_results: Optional[dict[str, list[Any]]] = None
    posthoc_hooks: Optional[list[Any]] = None
    log_dir: Optional[str] = None
    run_log_file: Optional[str] = None

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
    selection_metric_spec: dict[str, Any] = field(default_factory=dict)

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
            "report_evaluator": None if self.report_evaluator is None else _snapshot_object(self.report_evaluator),
            "outer_report_results": copy.deepcopy(self.outer_report_results),
            "outer_posthoc_results": copy.deepcopy(self.outer_posthoc_results),
            "posthoc_hooks": copy.deepcopy(self.posthoc_hooks),
            "log_dir": self.log_dir,
            "run_log_file": self.run_log_file,
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
            "selection_metric_spec": copy.deepcopy(self.selection_metric_spec),
        }

        if include_specs_repr:
            out["base_model_spec_repr"] = None if self.base_model_spec is None else repr(self.base_model_spec)
            out["base_trainer_spec_repr"] = None if self.base_trainer_spec is None else repr(self.base_trainer_spec)

        out["base_model_spec"] = None if self.base_model_spec is None else _snapshot_object(self.base_model_spec)
        out["base_trainer_spec"] = None if self.base_trainer_spec is None else _snapshot_object(self.base_trainer_spec)

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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NestedOptunaSearchCVResult":
        return cls(
            outer_results=[
                OuterFoldResult.from_dict(outer_payload)
                for outer_payload in payload.get("outer_results", [])
            ],
            base_model_spec=copy.deepcopy(payload.get("base_model_spec")),
            base_trainer_spec=copy.deepcopy(payload.get("base_trainer_spec")),
            parameter_grid=copy.deepcopy(payload.get("parameter_grid")),
            report_evaluator=copy.deepcopy(payload.get("report_evaluator")),
            outer_report_results=copy.deepcopy(payload.get("outer_report_results")),
            outer_posthoc_results=copy.deepcopy(payload.get("outer_posthoc_results")),
            posthoc_hooks=copy.deepcopy(payload.get("posthoc_hooks")),
            log_dir=payload.get("log_dir"),
            run_log_file=payload.get("run_log_file"),
            outer_splitter_name=payload.get("outer_splitter_name", ""),
            inner_splitter_name=payload.get("inner_splitter_name", ""),
            k_outer=int(payload.get("k_outer", 0)),
            k_inner=int(payload.get("k_inner", 0)),
            shuffle_outer=bool(payload.get("shuffle_outer", False)),
            shuffle_inner=bool(payload.get("shuffle_inner", False)),
            random_state=payload.get("random_state"),
            n_trials=int(payload.get("n_trials", 0)),
            max_trial_attempts=int(payload.get("max_trial_attempts", 0)),
            calibrate=bool(payload.get("calibrate", True)),
            final_model_dir=payload.get("final_model_dir"),
            keep_final_model_state_dict_cpu=bool(payload.get("keep_final_model_state_dict_cpu", True)),
            selection_metric_name=payload.get("selection_metric_name", ""),
            selection_metric_direction=payload.get("selection_metric_direction", "maximize"),
            selection_metric_spec=copy.deepcopy(payload.get("selection_metric_spec", {})),
        )

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterator, Literal, Optional
import copy
import math
import os

import torch
from torch import Tensor

import optuna

from torchkit.distributed import DDPStrategy
from torchkit.models.Model._model import TorchkitModel
from torchkit.objectives import Objective, MultitaskObjective
from torchkit.evaluate.select.bundle import BundleSelectorEvaluator, SelectorEvaluator
from torchkit.evaluate.select._selector_evaluator import MetricDirection
from torchkit.train._event_log import JsonlEventLogger, default_log_dir


def _move_to_device(x: Any, device: torch.device | str) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_move_to_device(v, device) for v in x]
        return type(x)(t) if isinstance(x, tuple) else t
    return x


def _scheduler_expects_metric(sched: object) -> bool:
    return isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)


def _scheduler_cls_expects_metric(
    scheduler_cls: Optional[type[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau]],
) -> bool:
    if scheduler_cls is None:
        return False
    return issubclass(scheduler_cls, torch.optim.lr_scheduler.ReduceLROnPlateau)


@dataclass(frozen=False)
class TrainerConfig:
    device: Optional[str | torch.device] = None
    random_seed: Optional[int] = None

    backbone_kwargs: Optional[dict[str, Any]] = None
    head_kwargs: Optional[dict[str, dict[str, Any]]] = None

    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})

    scheduler_cls: Optional[type[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau]] = None
    scheduler_kwargs: Optional[dict[str, Any]] = None
    scheduler_monitor: Optional[Literal["val_loss", "selector_metric"]] = None

    use_amp: bool = False
    grad_clip_norm: Optional[float] = None

    max_epochs: int = 50
    validate_every: int = 1
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None

    optuna_report_interval: int = 1

    model_initial_state_path: Optional[str] = None

    keep_history_on_reset: bool = False


@dataclass(frozen=False)
class TrainerState:
    epoch: int = 0

    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None

    epochs_since_improvement: int = 0

    train_logs: list[dict[str, Any]] = field(default_factory=list)
    val_logs: list[dict[str, Any]] = field(default_factory=list)

    best_state_dict_cpu: Optional[dict[str, torch.Tensor]] = None
    oof_logits: dict[str, torch.Tensor] = field(default_factory=dict)
    oof_targets: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    train_log: dict[str, Any]
    val_log: Optional[dict[str, Any]]
    did_validate: bool
    best_epoch: Optional[int]
    best_metric: Optional[float]
    epochs_since_improvement: int
    selection_score: Optional[float] = None


@dataclass(frozen=True)
class EpochControl:
    stop_training: bool = False
    prune_trial: bool = False
    report_value: Optional[float] = None
    suppress_default_report: bool = False


class Trainer:
    @staticmethod
    def _resolve_runtime_device(
        configured_device: Optional[str | torch.device],
        *,
        distributed_strategy: Optional[DDPStrategy],
        fallback_model: TorchkitModel,
    ) -> torch.device:
        if configured_device is not None:
            resolved = torch.device(configured_device)
            if (
                distributed_strategy is not None
                and distributed_strategy.is_enabled
                and resolved.type == "cuda"
                and resolved.index is None
            ):
                return distributed_strategy.device
            return resolved

        if distributed_strategy is not None and distributed_strategy.is_enabled:
            return distributed_strategy.device

        try:
            return next(fallback_model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def __init__(
        self,
        model: TorchkitModel,
        objective: Objective,
        *,
        selector_evaluator: Optional[BundleSelectorEvaluator] = None,
        config: Optional[TrainerConfig] = None,
        logging: bool = False,
        distributed_strategy: Optional[DDPStrategy] = None,
    ):
        self.model = model
        self.objective = objective
        self.selector_evaluator = selector_evaluator
        self.logging = bool(logging)
        self.distributed_strategy = distributed_strategy
        self._fit_model: torch.nn.Module = model

        self.config: TrainerConfig = copy.deepcopy(config) if config is not None else TrainerConfig()

        self.device = self._resolve_runtime_device(
            self.config.device,
            distributed_strategy=self.distributed_strategy,
            fallback_model=self.model,
        )

        self._initial_state_dict_cpu: dict[str, torch.Tensor] = self._load_initial_state_dict_cpu(
            model=self.model,
            model_initial_state_path=self.config.model_initial_state_path,
        )

        self.optimizer: torch.optim.Optimizer
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
        self._scaler: Optional[torch.amp.GradScaler]

        self.state = TrainerState()
        self._fit_called_at_least_once = False

        self._rebuild_stateful_objects_from_config(rebuild_optimizer=True, rebuild_scheduler=True, rebuild_scaler=True)

        self._base_config: TrainerConfig = copy.deepcopy(self.config)

        self.history: list[dict[str, Any]] = []
        self.log_dir: Optional[str] = None
        self.log_file: Optional[str] = None
        self._event_logger: Optional[JsonlEventLogger] = None

    def _set_event_logger(self, logger: Optional[JsonlEventLogger]) -> None:
        self._event_logger = logger
        if logger is not None:
            self.log_file = logger.path
            self.log_dir = os.path.dirname(logger.path)

    def _ensure_event_logger(self) -> Optional[JsonlEventLogger]:
        if (
            self.distributed_strategy is not None
            and self.distributed_strategy.is_enabled
            and not self.distributed_strategy.is_main_process
        ):
            return None
        if self._event_logger is not None:
            return self._event_logger
        if not self.logging:
            return None

        self.log_dir = default_log_dir(prefix="trainer")
        self.log_file = os.path.join(self.log_dir, "trainer.log.jsonl")
        self._event_logger = JsonlEventLogger(
            self.log_file,
            scope="trainer",
            echo_console=True,
        )
        return self._event_logger

    @staticmethod
    def _format_metric(value: Any) -> str:
        if value is None:
            return "None"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return str(value)
        fv = float(value)
        if not math.isfinite(fv):
            return str(fv)
        return f"{fv:.6f}"

    def _current_lr(self) -> float | list[float] | None:
        if not hasattr(self, "optimizer") or self.optimizer is None:
            return None
        lrs = [
            float(pg.get("lr"))
            for pg in getattr(self.optimizer, "param_groups", [])
            if "lr" in pg
        ]
        if not lrs:
            return None
        if len(lrs) == 1:
            return lrs[0]
        return lrs

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        d = asdict(self.config)
        return copy.deepcopy(d) if deep else d

    def set_params(self, **params: Any) -> "Trainer":
        if not params:
            return self

        rebuild_optimizer = False
        rebuild_scheduler = False
        rebuild_scaler = False
        update_device = False

        for k, v in params.items():
            if not hasattr(self.config, k):
                raise ValueError(f"Unknown TrainerConfig parameter: {k}")

            setattr(self.config, k, v)

            if k in ("optimizer_cls", "optimizer_kwargs"):
                rebuild_optimizer = True
                rebuild_scheduler = True
            if k in ("scheduler_cls", "scheduler_kwargs"):
                rebuild_scheduler = True
            if k in ("use_amp",):
                rebuild_scaler = True
            if k in ("device",):
                update_device = True
            if k in ("model_initial_state_path",):
                self._initial_state_dict_cpu = self._load_initial_state_dict_cpu(
                    model=self.model,
                    model_initial_state_path=self.config.model_initial_state_path,
                )

        if update_device:
            if self.config.device is None:
                raise ValueError("device cannot be None in set_params; pass a device or call reset_config().")
            self.device = self._resolve_runtime_device(
                self.config.device,
                distributed_strategy=self.distributed_strategy,
                fallback_model=self.model,
            )
            self.model.to(self.device)

        if rebuild_optimizer or rebuild_scheduler or rebuild_scaler:
            self._rebuild_stateful_objects_from_config(
                rebuild_optimizer=rebuild_optimizer,
                rebuild_scheduler=rebuild_scheduler,
                rebuild_scaler=rebuild_scaler,
            )

        return self

    def reset_config(self) -> None:
        self.config = copy.deepcopy(self._base_config)
        self.device = self._resolve_runtime_device(
            self.config.device,
            distributed_strategy=self.distributed_strategy,
            fallback_model=self.model,
        )
        self.model.to(self.device)
        self._rebuild_stateful_objects_from_config()

    def reset_state(self, *, reset_config: bool = False, clear_history: Optional[bool] = None) -> None:
        if reset_config:
            self.reset_config()
        else:
            self._rebuild_stateful_objects_from_config()

        self._restore_model_weights_from_cpu_snapshot()

        self.state = TrainerState()
        self._fit_called_at_least_once = False

        if clear_history is None:
            clear_history = not self.config.keep_history_on_reset
        if clear_history:
            self.history.clear()

    def detach_model(self) -> TorchkitModel:
        return self.model

    @staticmethod
    def _merge_epoch_controls(
        left: Optional[EpochControl],
        right: Optional[EpochControl],
    ) -> Optional[EpochControl]:
        if left is None:
            return right
        if right is None:
            return left
        return EpochControl(
            stop_training=bool(left.stop_training or right.stop_training),
            prune_trial=bool(left.prune_trial or right.prune_trial),
            report_value=right.report_value if right.report_value is not None else left.report_value,
            suppress_default_report=bool(left.suppress_default_report or right.suppress_default_report),
        )

    @staticmethod
    def _validate_epoch_control(
        control: Optional[EpochControl],
        *,
        hook_name: str,
    ) -> Optional[EpochControl]:
        if control is None:
            return None
        if not isinstance(control, EpochControl):
            raise TypeError(f"{hook_name} must return EpochControl or None, got {type(control).__name__}.")
        return control

    @staticmethod
    def _default_report_value_for_epoch(epoch_result: EpochResult) -> float:
        if epoch_result.did_validate and epoch_result.val_log is not None:
            score = epoch_result.val_log.get("__selection_score__", None)
            if score is None:
                score = -float(epoch_result.val_log["val_loss"])
            return float(score)
        return -float(epoch_result.train_log["train_loss"])

    def _distributed_enabled(self) -> bool:
        return bool(self.distributed_strategy is not None and self.distributed_strategy.is_enabled)

    def _gather_object(self, obj: Any) -> list[Any]:
        if not self._distributed_enabled():
            return [obj]
        assert self.distributed_strategy is not None
        return self.distributed_strategy.all_gather_object(obj)

    @staticmethod
    def _sum_float_dicts(dicts: list[dict[str, float]]) -> dict[str, float]:
        merged: dict[str, float] = {}
        for part in dicts:
            for key, value in part.items():
                merged[key] = merged.get(key, 0.0) + float(value)
        return merged

    @staticmethod
    def _sum_nested_float_dicts(dicts: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
        merged: dict[str, dict[str, float]] = {}
        for part in dicts:
            for outer_key, inner_dict in part.items():
                slot = merged.setdefault(outer_key, {})
                for inner_key, value in inner_dict.items():
                    slot[inner_key] = slot.get(inner_key, 0.0) + float(value)
        return merged

    @staticmethod
    def _concat_tensor_fragments(
        fragments: list[torch.Tensor],
        *,
        key: str,
    ) -> torch.Tensor:
        if len(fragments) == 0:
            raise ValueError(f"Empty tensor fragments for key {key!r} (unexpected).")
        if len(fragments) == 1:
            return fragments[0]
        if fragments[0].ndim == 0:
            return torch.stack(fragments, dim=0)
        return torch.cat(fragments, dim=0)

    def _train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        train_model = self._fit_model
        train_model.train()

        device = self.device
        if not isinstance(device, torch.device):
            device = torch.device(device)

        total_loss = 0.0
        num_batches = 0

        per_objective_sum_loss: dict[str, float] = {}

        for batch in train_loader:
            if not isinstance(batch, dict):
                raise TypeError(
                    f"Expected batch as dict[str, Any], got {type(batch)}. "
                    "Use a collate_fn that returns a dict. Names are required for routing inputs to the objective."
                )
            if "x" not in batch.keys():
                raise KeyError(
                    f"Expected batch to contain the 'x' key for the primary model input, but got keys: {list(batch.keys())}. "
                    "Use a `collate_fn` that puts model inputs under 'x'. This is an enforced convention for this library."
                )

            batch = _move_to_device(batch, device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.config.use_amp:
                with torch.autocast(device_type=device.type):
                    model_out = train_model(batch, backbone_kwargs=self.config.backbone_kwargs, head_kwargs=self.config.head_kwargs)
                    objective_in = dict(model_out)
                    objective_in["batch"] = batch
                    loss: Tensor = self.objective(inputs=objective_in)

                self._scaler.scale(loss).backward()
                if self.config.grad_clip_norm is not None:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                model_out = train_model(batch, backbone_kwargs=self.config.backbone_kwargs, head_kwargs=self.config.head_kwargs)
                objective_in = dict(model_out)
                objective_in["batch"] = batch
                loss = self.objective(inputs=objective_in)

                loss.backward()
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
                self.optimizer.step()

            total_loss += float(loss.detach().item())
            num_batches += 1

            if isinstance(self.objective, MultitaskObjective):
                for o, l in self.objective.per_objective_loss.items():
                    fl = float(l.detach().item()) if isinstance(l, torch.Tensor) else float(l)
                    per_objective_sum_loss[o] = per_objective_sum_loss.get(o, 0.0) + fl

        gathered_summaries = self._gather_object(
            {
                "total_loss": float(total_loss),
                "num_batches": int(num_batches),
                "per_objective_sum_loss": copy.deepcopy(per_objective_sum_loss),
            }
        )
        total_loss = sum(float(part["total_loss"]) for part in gathered_summaries)
        num_batches = sum(int(part["num_batches"]) for part in gathered_summaries)
        per_objective_sum_loss = self._sum_float_dicts(
            [dict(part["per_objective_sum_loss"]) for part in gathered_summaries]
        )

        if num_batches == 0:
            raise ValueError("train_loader produced 0 batches.")

        epoch_log = {
            "epoch": epoch,
            "train_loss": total_loss / num_batches,
        }
        if isinstance(self.objective, MultitaskObjective):
            for o, s in per_objective_sum_loss.items():
                epoch_log[f"train_loss/{o}"] = s / num_batches

        self.state.train_logs.append(epoch_log)
        return epoch_log

    def _selector_monitor_direction(self) -> MetricDirection:
        selector_bundle = getattr(self, "selector_evaluator", None)
        if selector_bundle is None:
            raise ValueError(
                "scheduler_monitor='selector_metric' requires `selector_evaluator` to be provided."
            )
        return "maximize"

    def _validate_scheduler_monitor_config(self) -> None:
        if not _scheduler_cls_expects_metric(self.config.scheduler_cls):
            return

        monitor = self.config.scheduler_monitor
        if monitor is None:
            raise ValueError(
                "ReduceLROnPlateau requires `scheduler_monitor` to be set to "
                "'val_loss' or 'selector_metric'."
            )
        if monitor == "selector_metric" and self.selector_evaluator is None:
            raise ValueError(
                "scheduler_monitor='selector_metric' requires `selector_evaluator` to be provided."
            )

    def _resolve_scheduler_monitor_value(self, val_log: dict[str, Any]) -> float:
        import math
        import numbers

        monitor = self.config.scheduler_monitor
        if monitor == "val_loss":
            value = val_log.get("val_loss", None)
            direction: MetricDirection = "minimize"
        elif monitor == "selector_metric":
            if "val/primary" not in val_log:
                raise ValueError(
                    "scheduler_monitor='selector_metric' requires a finite selector metric, "
                    "but validation did not produce 'val/primary'."
                )
            value = val_log["val/primary"]
            direction = self._selector_monitor_direction()
        else:
            raise ValueError(
                f"Unsupported scheduler_monitor {monitor!r}. Expected 'val_loss' or 'selector_metric'."
            )

        if isinstance(value, bool) or not isinstance(value, numbers.Number):
            raise TypeError(
                f"Scheduler monitor {monitor!r} must resolve to a numeric scalar, got {type(value).__name__}."
            )

        resolved_value = float(value)
        if not math.isfinite(resolved_value):
            raise ValueError(
                f"Scheduler monitor {monitor!r} resolved to non-finite value {resolved_value}."
            )

        scheduler_mode = getattr(self.scheduler, "mode", "min")
        if scheduler_mode not in ("min", "max"):
            raise ValueError(f"Unsupported ReduceLROnPlateau mode {scheduler_mode!r}.")

        if direction == "minimize":
            return resolved_value if scheduler_mode == "min" else -resolved_value
        if direction == "maximize":
            return -resolved_value if scheduler_mode == "min" else resolved_value
        raise ValueError(f"Unsupported monitor direction {direction!r}.")

    def _validate_one_epoch(
        self,
        val_loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        import math
        import numbers
        from collections import defaultdict

        self.model.eval()

        device = self.device
        if not isinstance(device, torch.device):
            device = torch.device(device)

        def _infer_batch_size(batch_dict: dict[str, Any]) -> int:
            x = batch_dict.get("x", None)
            if x is None:
                raise KeyError("Expected batch to contain a Tensor 'x' key for primary model input, but it was not found. Cannot infer batch size.")
            if not torch.is_tensor(x):
                raise TypeError(f"'x' is supposed to be a Tensor for primary model input, but got {type(x).__name__}. Cannot infer batch size.")
            if x.ndim == 0:
                raise ValueError("batch['x'] is scalar; cannot infer batch size.")
            return int(x.shape[0])

        def _set_by_path(root: dict[str, Any], path: str, value: Any) -> None:
            cur = root
            parts = [p for p in path.split("/") if p]
            if not parts:
                raise ValueError(f"Invalid empty path: {path!r}")
            for part in parts[:-1]:
                nxt = cur.get(part, None)
                if nxt is None:
                    nxt = {}
                    cur[part] = nxt
                if not isinstance(nxt, dict):
                    raise TypeError(f"Cannot set into non-dict at path segment {part!r} for full path {path!r}.")
                cur = nxt
            cur[parts[-1]] = value

        def _append_cached(cache: dict[str, list[torch.Tensor]], key: str, tensor: torch.Tensor) -> None:
            cache[key].append(tensor.detach().cpu())

        def _cat_cached_list(ts: list[torch.Tensor], key: str) -> torch.Tensor:
            return self._concat_tensor_fragments(ts, key=key)

        def _is_finite_number(x: Any) -> bool:
            if x is None:
                return False
            if isinstance(x, bool) or not isinstance(x, numbers.Number):
                return False
            fx = float(x)
            return math.isfinite(fx)

        def _selector_signed_value(selector: SelectorEvaluator, raw_value: float) -> float:
            if selector.direction == "maximize":
                signed = raw_value
            elif selector.direction == "minimize":
                signed = -raw_value
            else:
                raise ValueError(f"Unsupported selector direction {selector.direction!r}.")
            return float(selector.weight) * float(signed)

        def _accumulate_selector_components(
            accumulator: dict[str, dict[str, float]],
            components: dict[str, dict[str, Any]],
            *,
            batch_weight: float,
        ) -> None:
            for comp_name, comp_vals in components.items():
                slot = accumulator.setdefault(
                    comp_name,
                    {"raw": 0.0, "signed": 0.0, "weighted": 0.0},
                )
                slot["raw"] += float(comp_vals["raw"]) * batch_weight
                slot["signed"] += float(comp_vals["signed"]) * batch_weight
                slot["weighted"] += float(comp_vals["weighted"]) * batch_weight

        def _log_selector_components(
            epoch_log: dict[str, Any],
            *,
            prefix: str,
            components: dict[str, dict[str, Any]],
        ) -> None:
            for comp_name, comp_vals in components.items():
                epoch_log[f"{prefix}/{comp_name}/raw"] = float(comp_vals["raw"])
                epoch_log[f"{prefix}/{comp_name}/signed"] = float(comp_vals["signed"])
                epoch_log[f"{prefix}/{comp_name}/weighted"] = float(comp_vals["weighted"])

        def _find_task_targets(task: str, batch_dict: dict[str, Any]) -> torch.Tensor:
            if task in batch_dict and isinstance(batch_dict[task], dict):
                td = batch_dict[task]
                for k in ("y", "target", "targets", "label", "labels"):
                    v = td.get(k, None)
                    if torch.is_tensor(v):
                        return v
            for k in (f"{task}/y", f"{task}/target", f"{task}/targets", f"{task}/label", f"{task}/labels"):
                v = batch_dict.get(k, None)
                if torch.is_tensor(v):
                    return v
            for k in ("y", "target", "targets", "label", "labels"):
                v = batch_dict.get(k, None)
                if torch.is_tensor(v):
                    return v

            raise KeyError(
                f"Could not find calibration targets for task {task!r} in batch. "
                f"Available top-level batch keys: {list(batch_dict.keys())}. "
                f"Expected one of: batch[{task!r}][y/target/targets/label/labels] or a global y/target."
            )

        batch_selector = None
        dataset_selector = None
        batch_required_keys: tuple[str, ...] = ()
        dataset_required_keys: tuple[str, ...] = ()

        if getattr(self, "selector_evaluator", None) is not None:
            batch_selector = self.selector_evaluator.batch_evaluator
            dataset_selector = self.selector_evaluator.dataset_evaluator
            if batch_selector is not None:
                batch_required_keys = tuple(batch_selector.required_keys)
            if dataset_selector is not None:
                dataset_required_keys = tuple(dataset_selector.required_keys)

        dataset_cache: dict[str, list[torch.Tensor]] = defaultdict(list)

        total_loss = 0.0
        num_batches = 0

        batch_selector_weight_sum = 0.0
        batch_selector_weighted_value_sum = 0.0
        batch_selector_component_sums: dict[str, dict[str, float]] = {}

        active_posthoc_tasks = tuple(getattr(self.model, "active_posthoc_output_names", set()) or ())
        oof_logits_cache: dict[str, list[torch.Tensor]] = defaultdict(list)
        oof_targets_cache: dict[str, list[torch.Tensor]] = defaultdict(list)

        with torch.no_grad():
            for batch in val_loader:
                if not isinstance(batch, dict):
                    raise TypeError(
                        f"Expected batch as dict[str, Any], got {type(batch)}. "
                        "Use a collate_fn that returns a dict. Names are required for routing."
                    )
                if "x" not in batch.keys():
                    raise KeyError(
                        f"Expected batch to contain the 'x' key for the primary model input, but got keys: {list(batch.keys())}. "
                        "Use a `collate_fn` that puts model inputs under 'x'. This is an enforced convention for this library."
                    )

                bs = _infer_batch_size(batch)
                batch = _move_to_device(batch, device)

                if self.config.use_amp:
                    with torch.autocast(device_type=device.type):
                        model_out = self.model.predict(
                            batch,
                            *self.model.active_head_names,
                            backbone_kwargs=self.config.backbone_kwargs,
                            head_kwargs=self.config.head_kwargs,
                            return_raw_head_outputs=True,
                        )
                        eval_in = dict(model_out)
                        eval_in["batch"] = batch
                        loss = self.objective(inputs=eval_in)
                else:
                    model_out = self.model.predict(
                        batch,
                        *self.model.active_head_names,
                        backbone_kwargs=self.config.backbone_kwargs,
                        head_kwargs=self.config.head_kwargs,
                        return_raw_head_outputs=True,
                    )
                    eval_in = dict(model_out)
                    eval_in["batch"] = batch
                    loss = self.objective(inputs=eval_in)

                total_loss += float(loss.detach().item())
                num_batches += 1

                if batch_selector is not None:
                    batch_primary, batch_components = batch_selector.compute(inputs=eval_in)
                    batch_primary_value = float(batch_primary.detach().cpu().item())
                    if _is_finite_number(batch_primary_value):
                        batch_selector_weighted_value_sum += float(batch_primary_value) * bs
                        batch_selector_weight_sum += bs
                        _accumulate_selector_components(
                            batch_selector_component_sums,
                            batch_components,
                            batch_weight=float(bs),
                        )

                if dataset_required_keys:
                    for key in dataset_required_keys:
                        val = dataset_selector.resolve(eval_in, key)
                        if not torch.is_tensor(val):
                            raise TypeError(
                                f"dataset selector required key {key!r} resolved to {type(val).__name__}, expected Tensor. "
                                "Fix your dataset/collate/evaluator keys."
                            )
                        _append_cached(dataset_cache, key, val)

                if active_posthoc_tasks:
                    for task in active_posthoc_tasks:
                        if task not in model_out:
                            continue
                        node = model_out[task]
                        if not isinstance(node, dict):
                            continue
                        logits = node.get("logits", None)
                        if not torch.is_tensor(logits):
                            raise KeyError(
                                f"Expected model_out[{task!r}]['logits'] Tensor for post-hoc OOF logging, got {type(logits).__name__}."
                            )

                        targets = _find_task_targets(task, batch)

                        if targets.ndim == 0:
                            raise ValueError("targets is scalar; cannot infer batch size.")
                        if logits.ndim == 0:
                            raise ValueError("logits is scalar; cannot infer batch size.")
                        if int(targets.shape[0]) != int(logits.shape[0]):
                            raise ValueError("targets and logits have incompatible batch sizes.")

                        oof_logits_cache[task].append(logits.detach().cpu())
                        oof_targets_cache[task].append(targets.detach().cpu())

        local_dataset_cache = {
            key: _cat_cached_list(ts, key=key)
            for key, ts in dataset_cache.items()
            if len(ts) > 0
        }
        local_oof_cache = {
            "logits": {
                task: _cat_cached_list(ts, key=f"{task}/logits")
                for task, ts in oof_logits_cache.items()
                if len(ts) > 0
            },
            "targets": {
                task: _cat_cached_list(ts, key=f"{task}/targets")
                for task, ts in oof_targets_cache.items()
                if len(ts) > 0
            },
        }
        gathered_summaries = self._gather_object(
            {
                "total_loss": float(total_loss),
                "num_batches": int(num_batches),
                "batch_selector_weight_sum": float(batch_selector_weight_sum),
                "batch_selector_weighted_value_sum": float(batch_selector_weighted_value_sum),
                "batch_selector_component_sums": copy.deepcopy(batch_selector_component_sums),
                "dataset_cache": local_dataset_cache,
                "oof_cache": local_oof_cache,
            }
        )

        total_loss = sum(float(part["total_loss"]) for part in gathered_summaries)
        num_batches = sum(int(part["num_batches"]) for part in gathered_summaries)
        batch_selector_weight_sum = sum(float(part["batch_selector_weight_sum"]) for part in gathered_summaries)
        batch_selector_weighted_value_sum = sum(float(part["batch_selector_weighted_value_sum"]) for part in gathered_summaries)
        batch_selector_component_sums = self._sum_nested_float_dicts(
            [dict(part["batch_selector_component_sums"]) for part in gathered_summaries]
        )

        dataset_cache = defaultdict(list)
        for part in gathered_summaries:
            for key, tensor in dict(part["dataset_cache"]).items():
                dataset_cache[key].append(tensor)

        gathered_oof_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
        gathered_oof_targets: dict[str, list[torch.Tensor]] = defaultdict(list)
        for part in gathered_summaries:
            oof_cache = dict(part["oof_cache"])
            for task, tensor in dict(oof_cache.get("logits", {})).items():
                gathered_oof_logits[task].append(tensor)
            for task, tensor in dict(oof_cache.get("targets", {})).items():
                gathered_oof_targets[task].append(tensor)

        if num_batches == 0:
            raise ValueError("val_loader produced 0 batches across all ranks.")

        epoch_log: dict[str, Any] = {
            "epoch": epoch,
            "val_loss": total_loss / num_batches,
        }

        batch_primary_value: Any = None
        batch_primary_components: dict[str, dict[str, float]] = {}
        if batch_selector is not None and batch_selector_weight_sum > 0.0:
            batch_primary_value = batch_selector_weighted_value_sum / batch_selector_weight_sum
            epoch_log[f"val_batch/{batch_selector.name}"] = float(batch_primary_value)

            for comp_name, comp_vals in batch_selector_component_sums.items():
                batch_primary_components[comp_name] = {
                    "raw": comp_vals["raw"] / batch_selector_weight_sum,
                    "signed": comp_vals["signed"] / batch_selector_weight_sum,
                    "weighted": comp_vals["weighted"] / batch_selector_weight_sum,
                }

            _log_selector_components(
                epoch_log,
                prefix="val_batch_components",
                components=batch_primary_components,
            )

        dataset_primary_value: Any = None
        dataset_primary_components: dict[str, dict[str, Any]] = {}
        if dataset_selector is not None:
            epoch_inputs: dict[str, Any] = {}
            for key, ts in dataset_cache.items():
                epoch_tensor = _cat_cached_list(ts, key=key)
                _set_by_path(epoch_inputs, key, epoch_tensor)

            dataset_primary, dataset_primary_components = dataset_selector.compute(inputs=epoch_inputs)
            dataset_primary_value = float(dataset_primary.detach().cpu().item())
            epoch_log[f"val/{dataset_selector.name}"] = dataset_primary_value

            _log_selector_components(
                epoch_log,
                prefix="val_components",
                components=dataset_primary_components,
            )

        selector_score = None
        if batch_selector is not None and _is_finite_number(batch_primary_value):
            selector_score = _selector_signed_value(batch_selector, float(batch_primary_value))

        if dataset_selector is not None and _is_finite_number(dataset_primary_value):
            dataset_score = _selector_signed_value(dataset_selector, float(dataset_primary_value))
            selector_score = dataset_score if selector_score is None else selector_score + dataset_score

        if selector_score is not None and _is_finite_number(selector_score):
            score = float(selector_score)
            best_raw_for_state = float(selector_score)
            best_metric_kind = "selector_primary"
            epoch_log["val/primary"] = float(selector_score)
        else:
            raw_loss = float(epoch_log["val_loss"])
            score = -raw_loss
            best_raw_for_state = raw_loss
            best_metric_kind = "val_loss"

        thr = self.config.early_stopping_threshold
        thr = 0.0 if thr is None else float(thr)

        best_score_prev = getattr(self.state, "_best_score", None)
        improved = (best_score_prev is None) or (score > float(best_score_prev) + thr)

        if improved:
            self.state.best_epoch = int(epoch)
            self.state.best_metric = float(best_raw_for_state)
            self.state.best_state_dict_cpu = self._get_model_state_dict_cpu()
            setattr(self.state, "_best_score", float(score))
            setattr(self.state, "_best_metric_kind", best_metric_kind)
            self.state.epochs_since_improvement = 0

            if active_posthoc_tasks:
                if not hasattr(self.state, "oof_logits"):
                    self.state.oof_logits = {}
                if not hasattr(self.state, "oof_targets"):
                    self.state.oof_targets = {}

                for task in active_posthoc_tasks:
                    ls = gathered_oof_logits.get(task, [])
                    ts = gathered_oof_targets.get(task, [])
                    if len(ls) == 0 or len(ts) == 0:
                        self.state.oof_logits[task] = torch.empty(0)
                        self.state.oof_targets[task] = torch.empty(0)
                        continue

                    log_cat = _cat_cached_list(ls, key=f"{task}/logits")
                    tgt_cat = _cat_cached_list(ts, key=f"{task}/targets")

                    self.state.oof_logits[task] = log_cat
                    self.state.oof_targets[task] = tgt_cat
        else:
            self.state.epochs_since_improvement += 1

        epoch_log["__selection_score__"] = float(score)

        self.state.val_logs.append(epoch_log)
        return epoch_log

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        *,
        reset_state: bool = True,
        trial: Optional[optuna.trial.Trial] = None,
        max_epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        optuna_report_interval: Optional[int] = None,
        validate_every: Optional[int] = None,
        after_validation: Optional[Callable[["Trainer", EpochResult], Optional[EpochControl]]] = None,
        on_epoch_end: Optional[Callable[["Trainer", EpochResult], Optional[EpochControl]]] = None,
    ) -> "Trainer":
        for _ in self.fit_iter(
            train_loader,
            val_loader,
            reset_state=reset_state,
            trial=trial,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            grad_clip_norm=grad_clip_norm,
            optuna_report_interval=optuna_report_interval,
            validate_every=validate_every,
            after_validation=after_validation,
            on_epoch_end=on_epoch_end,
        ):
            pass
        return self

    def fit_iter(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        *,
        reset_state: bool = True,
        trial: Optional[optuna.trial.Trial] = None,
        max_epochs: Optional[int] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        optuna_report_interval: Optional[int] = None,
        validate_every: Optional[int] = None,
        after_validation: Optional[Callable[["Trainer", EpochResult], Optional[EpochControl]]] = None,
        on_epoch_end: Optional[Callable[["Trainer", EpochResult], Optional[EpochControl]]] = None,
    ) -> Iterator[EpochResult]:
        old_grad_clip = self.config.grad_clip_norm
        old_pat = self.config.early_stopping_patience
        old_thr = self.config.early_stopping_threshold
        old_rep = self.config.optuna_report_interval
        old_validate_every = self.config.validate_every
        strategy = self.distributed_strategy
        if strategy is not None and strategy.is_enabled:
            strategy.initialize()
            self.device = self._resolve_runtime_device(
                self.config.device,
                distributed_strategy=strategy,
                fallback_model=self.model,
            )
        logger = self._ensure_event_logger()

        if grad_clip_norm is not None:
            self.config.grad_clip_norm = grad_clip_norm
        if early_stopping_patience is not None:
            self.config.early_stopping_patience = early_stopping_patience
        if early_stopping_threshold is not None:
            self.config.early_stopping_threshold = early_stopping_threshold
        if optuna_report_interval is not None:
            self.config.optuna_report_interval = optuna_report_interval
        if validate_every is not None:
            self.config.validate_every = validate_every

        try:
            if reset_state:
                self.reset_state(reset_config=False)

            if self.config.random_seed is not None:
                s = int(self.config.random_seed)
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)

            device = self.device
            if not isinstance(device, torch.device):
                device = torch.device(device)
            self.device = device
            self.model.to(device)
            self._fit_model = self.model
            if strategy is not None and strategy.is_enabled:
                self._fit_model = strategy.wrap_model(self.model)

            run_max_epochs = int(max_epochs) if max_epochs is not None else int(self.config.max_epochs)
            if run_max_epochs <= 0:
                raise ValueError(f"max_epochs must be > 0, got {run_max_epochs}.")

            run_validate_every = int(self.config.validate_every)
            if run_validate_every <= 0:
                raise ValueError(f"validate_every must be > 0, got {run_validate_every}.")

            patience = self.config.early_stopping_patience
            patience = None if patience is None else int(patience)
            if patience is not None and patience < 0:
                raise ValueError(f"early_stopping_patience must be >=0 or None, got {patience}.")

            report_every = int(self.config.optuna_report_interval) if self.config.optuna_report_interval is not None else 1
            if report_every <= 0:
                report_every = 1

            self._fit_called_at_least_once = True

            if logger is not None:
                logger.emit(
                    "trainer_fit_start",
                    payload={
                        "max_epochs": run_max_epochs,
                        "has_val_loader": val_loader is not None,
                        "device": str(device),
                        "use_amp": self.config.use_amp,
                        "scheduler_monitor": self.config.scheduler_monitor,
                        "validate_every": run_validate_every,
                        "early_stopping_patience": patience,
                        "early_stopping_threshold": self.config.early_stopping_threshold,
                    },
                    message=(
                        f"Trainer fit started: max_epochs={run_max_epochs}, "
                        f"has_val_loader={val_loader is not None}, device={device}. "
                        f"Logging to {logger.path}."
                    ),
                )

            for ep in range(1, run_max_epochs + 1):
                self.state.epoch = ep
                if strategy is not None and strategy.is_enabled:
                    strategy.set_epoch(train_loader, ep)
                    if val_loader is not None:
                        strategy.set_epoch(val_loader, ep)

                train_log = self._train_one_epoch(train_loader, epoch=ep)

                if self.scheduler is not None and not _scheduler_expects_metric(self.scheduler):
                    self.scheduler.step()

                should_validate = (
                    val_loader is not None
                    and (((ep % run_validate_every) == 0) or (ep == run_max_epochs))
                )

                val_log: Optional[dict[str, Any]] = None
                new_best_stored = False
                stop_from_patience = False

                if should_validate and val_loader is not None:
                    prev_best_epoch = self.state.best_epoch
                    val_log = self._validate_one_epoch(val_loader, epoch=ep)
                    new_best_stored = self.state.best_epoch == ep and self.state.best_epoch != prev_best_epoch

                    if self.scheduler is not None and _scheduler_expects_metric(self.scheduler):
                        self.scheduler.step(self._resolve_scheduler_monitor_value(val_log))

                    if patience is not None:
                        if self.state.epochs_since_improvement >= patience:
                            stop_from_patience = True
                            if logger is not None:
                                logger.emit(
                                    "trainer_early_stop",
                                    payload={
                                        "epoch": ep,
                                        "best_epoch": self.state.best_epoch,
                                        "best_metric": self.state.best_metric,
                                        "epochs_since_improvement": self.state.epochs_since_improvement,
                                    },
                                    message=(
                                        f"Early stopping triggered at epoch {ep}. "
                                        f"Best epoch={self.state.best_epoch}, "
                                        f"best_metric={self._format_metric(self.state.best_metric)}."
                                    ),
                                )

                    if logger is not None:
                        metric_key = "val/primary" if "val/primary" in val_log else "val_loss"
                        selection_score = val_log.get("__selection_score__", None)
                        msg = (
                            f"Epoch {ep}/{run_max_epochs}: "
                            f"train_loss={self._format_metric(train_log.get('train_loss'))}, "
                            f"{metric_key}={self._format_metric(val_log.get(metric_key))}, "
                            f"best_metric={self._format_metric(self.state.best_metric)}"
                        )
                        if new_best_stored:
                            msg += ", new best snapshot stored in memory"
                        logger.emit(
                            "trainer_epoch_end",
                            payload={
                                "epoch": ep,
                                "max_epochs": run_max_epochs,
                                "train_log": copy.deepcopy(train_log),
                                "val_log": copy.deepcopy(val_log),
                                "best_epoch": self.state.best_epoch,
                                "best_metric": self.state.best_metric,
                                "epochs_since_improvement": self.state.epochs_since_improvement,
                                "selection_score": selection_score,
                                "lr": self._current_lr(),
                                "new_best_snapshot_in_memory": new_best_stored,
                            },
                            message=msg,
                        )

                else:
                    if logger is not None:
                        logger.emit(
                            "trainer_epoch_end",
                            payload={
                                "epoch": ep,
                                "max_epochs": run_max_epochs,
                                "train_log": copy.deepcopy(train_log),
                                "val_log": None,
                                "best_epoch": self.state.best_epoch,
                                "best_metric": self.state.best_metric,
                                "epochs_since_improvement": self.state.epochs_since_improvement,
                                "selection_score": None,
                                "lr": self._current_lr(),
                                "new_best_snapshot_in_memory": False,
                            },
                            message=(
                                f"Epoch {ep}/{run_max_epochs}: "
                                f"train_loss={self._format_metric(train_log.get('train_loss'))}."
                            ),
                        )

                selection_score = None
                if val_log is not None:
                    score_raw = val_log.get("__selection_score__", None)
                    selection_score = None if score_raw is None else float(score_raw)

                epoch_result = EpochResult(
                    epoch=ep,
                    train_log=copy.deepcopy(train_log),
                    val_log=copy.deepcopy(val_log),
                    did_validate=bool(should_validate),
                    best_epoch=self.state.best_epoch,
                    best_metric=self.state.best_metric,
                    epochs_since_improvement=self.state.epochs_since_improvement,
                    selection_score=selection_score,
                )

                control = None
                if should_validate and after_validation is not None:
                    control = self._merge_epoch_controls(
                        control,
                        self._validate_epoch_control(
                            after_validation(self, epoch_result),
                            hook_name="after_validation",
                        ),
                    )
                if on_epoch_end is not None:
                    control = self._merge_epoch_controls(
                        control,
                        self._validate_epoch_control(
                            on_epoch_end(self, epoch_result),
                            hook_name="on_epoch_end",
                        ),
                    )

                if (trial is not None or self._distributed_enabled()) and (ep % report_every == 0):
                    report_value = None
                    if control is not None and control.report_value is not None:
                        report_value = float(control.report_value)
                    elif (val_loader is None or epoch_result.did_validate) and not (
                        control is not None and control.suppress_default_report
                    ):
                        report_value = self._default_report_value_for_epoch(epoch_result)
                    if report_value is not None:
                        self.maybe_report_to_trial(trial, value=report_value, step=ep)

                if control is not None and control.prune_trial:
                    raise optuna.TrialPruned()

                yield epoch_result

                if stop_from_patience or (control is not None and control.stop_training):
                    break

            self.history.append(
                {
                    "best_epoch": self.state.best_epoch,
                    "best_metric": self.state.best_metric,
                    "train_last": (self.state.train_logs[-1] if self.state.train_logs else None),
                    "val_last": (self.state.val_logs[-1] if self.state.val_logs else None),
                }
            )

            if self.state.best_state_dict_cpu is not None:
                sd = {k: v.to(device, non_blocking=True) for k, v in self.state.best_state_dict_cpu.items()}
                self.model.load_state_dict(sd, strict=True)

            if logger is not None:
                logger.emit(
                    "trainer_fit_end",
                    payload={
                        "epochs_ran": self.state.epoch,
                        "best_epoch": self.state.best_epoch,
                        "best_metric": self.state.best_metric,
                        "best_snapshot_in_memory": self.state.best_state_dict_cpu is not None,
                        "final_train_log": copy.deepcopy(self.state.train_logs[-1]) if self.state.train_logs else None,
                        "final_val_log": copy.deepcopy(self.state.val_logs[-1]) if self.state.val_logs else None,
                    },
                    message=(
                        f"Trainer fit ended after {self.state.epoch} epochs. "
                        f"Best epoch={self.state.best_epoch}, "
                        f"best_metric={self._format_metric(self.state.best_metric)}."
                    ),
                )

        except Exception as e:
            if logger is not None:
                logger.emit(
                    "trainer_fit_exception",
                    payload={
                        "epoch": self.state.epoch if hasattr(self, "state") else None,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    message=f"Trainer fit failed with {type(e).__name__}: {e}",
                )
            raise
        finally:
            self._fit_model = self.model
            self.config.grad_clip_norm = old_grad_clip
            self.config.early_stopping_patience = old_pat
            self.config.early_stopping_threshold = old_thr
            self.config.optuna_report_interval = old_rep
            self.config.validate_every = old_validate_every

    def maybe_report_to_trial(
        self,
        trial: Optional[optuna.trial.Trial],
        *,
        value: float,
        step: int,
    ) -> None:
        if self._distributed_enabled():
            assert self.distributed_strategy is not None
            should_prune = False
            if self.distributed_strategy.is_main_process and trial is not None:
                trial.report(value, step)
                should_prune = bool(trial.should_prune())
            should_prune = bool(self.distributed_strategy.broadcast_object(should_prune, src=0))
            if should_prune:
                raise optuna.TrialPruned()
            return

        if trial is None:
            return
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    def _get_model_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _load_initial_state_dict_cpu(
        self,
        *,
        model: TorchkitModel,
        model_initial_state_path: Optional[str],
    ) -> dict[str, torch.Tensor]:
        if model_initial_state_path is None:
            return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        sd = torch.load(model_initial_state_path, map_location="cpu")
        if not isinstance(sd, dict):
            raise TypeError("Loaded initial state is not a state_dict (expected dict[str, Tensor]).")
        return sd

    def _restore_model_weights_from_cpu_snapshot(self) -> None:
        sd = {k: v.to(self.device, non_blocking=True) for k, v in self._initial_state_dict_cpu.items()}
        self.model.load_state_dict(sd, strict=True)

    def _rebuild_stateful_objects_from_config(
        self,
        *,
        rebuild_optimizer: bool = True,
        rebuild_scheduler: bool = True,
        rebuild_scaler: bool = True,
    ) -> None:
        self._validate_scheduler_monitor_config()

        if rebuild_optimizer:
            self.optimizer = self.config.optimizer_cls(self.model.parameters(), **self.config.optimizer_kwargs)

        if rebuild_scheduler:
            if self.config.scheduler_cls is None:
                self.scheduler = None
            else:
                self.scheduler = self.config.scheduler_cls(
                    self.optimizer, **(self.config.scheduler_kwargs or {})
                )

        if rebuild_scaler:
            device = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
            self._scaler = torch.amp.GradScaler(enabled=(self.config.use_amp and device.type == "cuda"))

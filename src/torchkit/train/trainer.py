from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional
import copy
import math
import os

import torch
from torch import Tensor

import optuna

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


class Trainer:
    def __init__(
        self,
        model: TorchkitModel,
        objective: Objective,
        *,
        selector_evaluator: Optional[BundleSelectorEvaluator] = None,
        config: Optional[TrainerConfig] = None,
        logging: bool = False,
    ):
        self.model = model
        self.objective = objective
        self.selector_evaluator = selector_evaluator
        self.logging = bool(logging)

        self.config: TrainerConfig = copy.deepcopy(config) if config is not None else TrainerConfig()

        resolved_device = self.config.device
        if resolved_device is None:
            try:
                resolved_device = next(model.parameters()).device
            except StopIteration:
                resolved_device = torch.device("cpu")
        self.device = resolved_device

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
            self.device = self.config.device
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
        self.device = self.config.device or self.device
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

    def _train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        self.model.train()

        device = self.device
        if not isinstance(device, torch.device):
            device = torch.device(device)

        total_loss = 0.0
        num_batches = 0

        if isinstance(self.objective, MultitaskObjective):
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
                    model_out = self.model(batch, backbone_kwargs=self.config.backbone_kwargs, head_kwargs=self.config.head_kwargs)
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
                model_out = self.model(batch, backbone_kwargs=self.config.backbone_kwargs, head_kwargs=self.config.head_kwargs)
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
            if len(ts) == 0:
                raise ValueError(f"Empty cache list for key {key!r} (unexpected).")
            if len(ts) == 1:
                return ts[0]
            if ts[0].ndim == 0:
                return torch.stack(ts, dim=0)
            return torch.cat(ts, dim=0)

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

        if num_batches == 0:
            raise ValueError("val_loader produced 0 batches.")

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
                    ls = oof_logits_cache.get(task, [])
                    ts = oof_targets_cache.get(task, [])
                    if len(ls) == 0 or len(ts) == 0:
                        self.state.oof_logits[task] = torch.empty(0)
                        self.state.oof_targets[task] = torch.empty(0)
                        continue

                    log_cat = torch.cat(ls, dim=0) if ls[0].ndim >= 1 else torch.stack(ls, dim=0)
                    tgt_cat = torch.cat(ts, dim=0) if ts[0].ndim >= 1 else torch.stack(ts, dim=0)

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
    ) -> "Trainer":
        old_grad_clip = self.config.grad_clip_norm
        old_pat = self.config.early_stopping_patience
        old_thr = self.config.early_stopping_threshold
        old_rep = self.config.optuna_report_interval
        logger = self._ensure_event_logger()

        if grad_clip_norm is not None:
            self.config.grad_clip_norm = grad_clip_norm
        if early_stopping_patience is not None:
            self.config.early_stopping_patience = early_stopping_patience
        if early_stopping_threshold is not None:
            self.config.early_stopping_threshold = early_stopping_threshold
        if optuna_report_interval is not None:
            self.config.optuna_report_interval = optuna_report_interval

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
            self.model.to(device)

            run_max_epochs = int(max_epochs) if max_epochs is not None else int(self.config.max_epochs)
            if run_max_epochs <= 0:
                raise ValueError(f"max_epochs must be > 0, got {run_max_epochs}.")

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

                train_log = self._train_one_epoch(train_loader, epoch=ep)

                if self.scheduler is not None and not _scheduler_expects_metric(self.scheduler):
                    self.scheduler.step()

                if val_loader is not None:
                    prev_best_epoch = self.state.best_epoch
                    val_log = self._validate_one_epoch(val_loader, epoch=ep)
                    new_best_stored = self.state.best_epoch == ep and self.state.best_epoch != prev_best_epoch

                    if self.scheduler is not None and _scheduler_expects_metric(self.scheduler):
                        self.scheduler.step(self._resolve_scheduler_monitor_value(val_log))

                    if trial is not None and (ep % report_every == 0):
                        score = val_log.get("__selection_score__", None)
                        if score is None:
                            score = -float(val_log["val_loss"])
                        self.maybe_report_to_trial(trial, value=float(score), step=ep)

                    if patience is not None:
                        if self.state.epochs_since_improvement >= patience:
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
                            break

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
                    if trial is not None and (ep % report_every == 0):
                        score = -float(train_log["train_loss"])
                        self.maybe_report_to_trial(trial, value=float(score), step=ep)

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

            return self

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
            self.config.grad_clip_norm = old_grad_clip
            self.config.early_stopping_patience = old_pat
            self.config.early_stopping_threshold = old_thr
            self.config.optuna_report_interval = old_rep

    def maybe_report_to_trial(
        self,
        trial: Optional[optuna.trial.Trial],
        *,
        value: float,
        step: int,
    ) -> None:
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

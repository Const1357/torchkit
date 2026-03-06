from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Literal
import copy

import torch
from torch import Tensor

import optuna

from torchkit.models.Model._model import TorchkitModel
from torchkit.objectives import Objective, MultitaskObjective
from torchkit.evaluate import Evaluator


MetricDirection = Literal["minimize", "maximize"]

# recursively walk nested structures and move leaf tensors to device
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


@dataclass(frozen=False)
class TrainerConfig:

    # Core ---
    device: Optional[str | torch.device] = None
    random_seed: Optional[int] = None

    # Forward-time arguments (backbone_kwargs, head_kwargs)
    backbone_kwargs: Optional[dict[str, Any]] = None
    head_kwargs: Optional[dict[str, dict[str,Any]]] = None

    # Optimization ---
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW
    optimizer_kwargs: dict[str, Any] = field(default_factory=lambda: {"lr": 1e-3})

    scheduler_cls: Optional[type[torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau]] = None
    scheduler_kwargs: Optional[dict[str, Any]] = None

    # AMP / stability ---
    use_amp: bool = False
    grad_clip_norm: Optional[float] = None

    # Training control ---
    max_epochs: int = 50
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None

    # Optuna reporting ---
    optuna_report_interval: int = 1  # report every N epochs

    # Checkpoint / initialization ---
    model_initial_state_path: Optional[str] = None  # if provided, load initial weights from here

    # Misc policy knobs ---
    keep_history_on_reset: bool = False


@dataclass(frozen=False)
class TrainerState:
    """Resettable internal state for a single run."""
    epoch: int = 0

    best_metric: Optional[float] = None # evaluator's primary, or val loss if no evaluator
    best_epoch: Optional[int] = None

    # could store early stopping counters etc.
    epochs_since_improvement: int = 0

    # simple per-epoch logs (optional; you can remove)
    train_logs: list[dict[str, Any]] = field(default_factory=list)
    val_logs: list[dict[str, Any]] = field(default_factory=list)

    # cache: updated for best epoch
    best_state_dict_cpu: Optional[dict[str, torch.Tensor]] = None
    oof_logits: dict[str, torch.Tensor] = field(default_factory=dict)   # task -> logits  for all val samples at best epoch
    oof_targets: dict[str, torch.Tensor] = field(default_factory=dict)  # task -> targets for all val samples at best epoch


class Trainer:
    """
    Trainer executes a single training run given current config.
    It supports Optuna by accepting a `trial` in `fit()` and providing pruning/reporting hooks,
    but it does NOT own an Optuna Study.
    """

    def __init__(
        self,
        model: TorchkitModel,
        objective: Objective,
        *,
        dataset_evaluator: Optional[Evaluator] = None,
        batch_evaluator: Optional[Evaluator] = None,
        config: Optional[TrainerConfig] = None,
    ):
        self.model = model
        self.objective = objective
        self.dataset_evaluator = dataset_evaluator
        self.batch_evaluator = batch_evaluator

        # Current mutable config
        self.config: TrainerConfig = copy.deepcopy(config) if config is not None else TrainerConfig()

        # Resolve device once (can be overridden later via set_params(device=...))
        resolved_device = self.config.device
        if resolved_device is None:
            try:
                resolved_device = next(model.parameters()).device
            except StopIteration:
                resolved_device = torch.device("cpu")
        self.device = resolved_device

        # ---- initial weights snapshot (reset point) ----
        self._initial_state_dict_cpu: dict[str, torch.Tensor] = self._load_initial_state_dict_cpu(
            model=self.model,
            model_initial_state_path=self.config.model_initial_state_path,
        )

        # ---- stateful objects (resettable) ----
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
        self._scaler: Optional[torch.amp.GradScaler]

        self.state = TrainerState()
        self._fit_called_at_least_once = False

        # Create optimizer/scheduler/scaler from config
        self._rebuild_stateful_objects_from_config(rebuild_optimizer=True, rebuild_scheduler=True, rebuild_scaler=True)

        # Snapshot the `reset target` config AFTER full init
        self._base_config: TrainerConfig = copy.deepcopy(self.config)

        # Optional non-resettable history (off by default)
        self.history: list[dict[str, Any]] = []

    # -------------------------
    # Public API
    # -------------------------

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """
        sklearn-style param getter.
        Returns a dict of TrainerConfig fields. (deep currently just copies nested dicts)
        """
        d = asdict(self.config)
        return copy.deepcopy(d) if deep else d

    def set_params(self, **params: Any) -> "Trainer":
        """
        sklearn-style param setter.
        Updates config fields; rebuilds optimizer/scheduler/scaler if relevant fields changed.
        """
        if not params:
            return self

        # Track whether we must rebuild stateful objects
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
                # scheduler depends on optimizer
                rebuild_scheduler = True
            if k in ("scheduler_cls", "scheduler_kwargs"):
                rebuild_scheduler = True
            if k in ("use_amp",):
                rebuild_scaler = True
            if k in ("device",):
                update_device = True
            if k in ("model_initial_state_path",):
                # Changing reset weights requires refreshing the snapshot
                self._initial_state_dict_cpu = self._load_initial_state_dict_cpu(
                    model=self.model,
                    model_initial_state_path=self.config.model_initial_state_path,
                )

        if update_device:
            if self.config.device is None:
                raise ValueError("device cannot be None in set_params; pass a device or call reset_config().")
            self.device = self.config.device
            self.model.to(self.device)

        # Rebuild stateful objects
        if rebuild_optimizer or rebuild_scheduler or rebuild_scaler:
            self._rebuild_stateful_objects_from_config(
                rebuild_optimizer=rebuild_optimizer,
                rebuild_scheduler=rebuild_scheduler,
                rebuild_scaler=rebuild_scaler,
            )

        return self

    def reset_config(self) -> None:
        """Reset config back to the snapshot taken at __init__."""
        self.config = copy.deepcopy(self._base_config)
        self.device = self.config.device or self.device
        self.model.to(self.device)
        self._rebuild_stateful_objects_from_config()

    def reset_state(self, *, reset_config: bool = False, clear_history: Optional[bool] = None) -> None:
        """
        Reset *run state* back to the reset point captured at initialization.

        - Restores model weights to the stored initial snapshot.
        - Rebuilds optimizer/scheduler/scaler from *current* config (or base config if reset_config=True).
        - Clears internal counters, best score, logs.

        `clear_history`: if None, follows config.keep_history_on_reset
        """
        if reset_config:
            self.reset_config()
        else:
            # rebuild optimizer/scheduler/scaler from current config (`reset_config` already rebuilds it => no need to do it twice)
            self._rebuild_stateful_objects_from_config()

        # restore model weights
        self._restore_model_weights_from_cpu_snapshot()


        # reset trainer internal state
        self.state = TrainerState()
        self._fit_called_at_least_once = False

        # optionally clear persistent history
        if clear_history is None:
            clear_history = not self.config.keep_history_on_reset
        if clear_history:
            self.history.clear()

    def detach_model(self) -> TorchkitModel:
        """
        Helper API: detach the model from the trainer (useful if you want to keep a trained model
        but discard trainer internals).
        """
        return self.model

    # -------------------------
    # Fit / epoch signatures (unimplemented)
    # -------------------------

    def _train_one_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        """
        Runs one epoch of training and returns logging dict.
        """
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
            
            batch: dict[str, Any] = _move_to_device(batch, device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.config.use_amp:
                with torch.autocast(device_type=device.type):
                    model_out = self.model(batch, backbone_kwargs=self.config.backbone_kwargs, head_kwargs=self.config.head_kwargs)
                    objective_in = dict(model_out)
                    objective_in["batch"] = batch
                    loss: Tensor = self.objective(inputs=objective_in)
                
                self._scaler.scale(loss).backward()
                # grad clipping under AMP: (unscale first)
                if self.config.grad_clip_norm is not None:
                    self._scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
                self._scaler.step(self.optimizer)
                self._scaler.update()

            else:
                model_out = self.model(batch, backbone_kwargs=self.config.backbone_kwargs, head_kwargs=self.config.head_kwargs)
                objective_in = dict(model_out)
                objective_in["batch"] = batch
                loss: Tensor = self.objective(inputs=objective_in)

                loss.backward()
                if self.config.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.grad_clip_norm)
                self.optimizer.step()
            
            # epoch-level stats/logs
            total_loss += float(loss.detach().item())
            num_batches += 1

            if isinstance(self.objective, MultitaskObjective):
                for o, l in self.objective.per_objective_loss.items():
                    fl = float(l.detach().item())
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

    def _validate_one_epoch(
        self,
        val_loader: torch.utils.data.DataLoader,
        *,
        epoch: int,
    ) -> dict[str, Any]:
        """
        Runs one epoch of validation and returns an epoch-level logging dict.

        Also:
        - Tracks "best epoch" according to dataset_evaluator primary (or val_loss if no evaluator)
        - If this epoch becomes the new best, stores OOF logits/targets for all active calibrators:
            self.state.oof_logits[task], self.state.oof_targets[task]
        where task in self.model.active_calibrator_names.
        """

        import math
        import numbers
        from collections import defaultdict

        self.model.eval()

        device = self.device
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # helpers ---
        def _infer_batch_size(batch_dict: dict[str, Any]) -> int:
            x = batch_dict.get("x", None)

            if x is None:
                raise KeyError("Expected batch to contain a Tensor 'x' key for primary model input, but it was not found. Cannot infer batch size.")

            if not torch.is_tensor(x):
                raise KeyError(f"'x' is supposed to be a Tensor for primary model input, but got {type(x).__name__}. Cannot infer batch size.")
            else:
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

        def _find_task_targets(task: str, batch_dict: dict[str, Any]) -> torch.Tensor:
            """
            Best-effort target discovery for calibration OOF storage.
            Supports:
            - batch[task] is a dict with keys in ('y','target','targets','label','labels')
            - or flat batch keys: f"{task}/y" etc (rare, but some collates do this)
            - or single-task fallback: batch['y'] / batch['target'] / batch['labels']
            """
            # nested under batch[task]
            if task in batch_dict and isinstance(batch_dict[task], dict):
                td = batch_dict[task]
                for k in ("y", "target", "targets", "label", "labels"):
                    v = td.get(k, None)
                    if torch.is_tensor(v):
                        return v
            # flat keys
            for k in (f"{task}/y", f"{task}/target", f"{task}/targets", f"{task}/label", f"{task}/labels"):
                v = batch_dict.get(k, None)
                if torch.is_tensor(v):
                    return v
            # single-task fallback
            for k in ("y", "target", "targets", "label", "labels"):
                v = batch_dict.get(k, None)
                if torch.is_tensor(v):
                    return v

            raise KeyError(
                f"Could not find calibration targets for task {task!r} in batch. "
                f"Available top-level batch keys: {list(batch_dict.keys())}. "
                f"Expected one of: batch[{task!r}][y/target/targets/label/labels] or a global y/target."
            )

        # setup caches / accumulators ---
        dataset_required_keys: tuple[str, ...] = ()
        if getattr(self, "dataset_evaluator", None) is not None:
            dataset_required_keys = tuple(self.dataset_evaluator.required_keys)

        dataset_cache: dict[str, list[torch.Tensor]] = defaultdict(list)

        batch_metric_sums: dict[str, float] = defaultdict(float)
        batch_metric_weight_sums: dict[str, float] = defaultdict(float)

        total_loss = 0.0
        num_batches = 0

        # --- OOF caches for calibrators (only committed if epoch becomes best) ---
        active_calib_tasks = tuple(getattr(self.model, "active_calibrator_names", set()) or ())
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

                # forward + loss
                if self.config.use_amp:
                    with torch.autocast(device_type=device.type):
                        model_out = self.model(
                            batch,
                            backbone_kwargs=self.config.backbone_kwargs,
                            head_kwargs=self.config.head_kwargs,
                        )
                        eval_in = dict(model_out)
                        eval_in["batch"] = batch
                        loss: torch.Tensor = self.objective(inputs=eval_in)
                else:
                    model_out = self.model(
                        batch,
                        backbone_kwargs=self.config.backbone_kwargs,
                        head_kwargs=self.config.head_kwargs,
                    )
                    eval_in = dict(model_out)
                    eval_in["batch"] = batch
                    loss = self.objective(inputs=eval_in)

                total_loss += float(loss.detach().item())
                num_batches += 1

                # batch-level evaluator (optional): scalar numeric metrics, allow None/NaN skipping
                if getattr(self, "batch_evaluator", None) is not None:
                    bm = self.batch_evaluator(inputs=eval_in)
                    if not isinstance(bm, dict):
                        raise TypeError(f"batch_evaluator must return dict[str, Any], got {type(bm).__name__}.")

                    for k, v in bm.items():
                        if v is None:
                            continue
                        if isinstance(v, bool):
                            raise TypeError(f"batch_evaluator metric {k!r} is bool; expected a numeric scalar or None.")
                        if not isinstance(v, numbers.Number):
                            raise TypeError(
                                f"batch_evaluator metric {k!r} must be a python number (float/int) or None for aggregation, "
                                f"got {type(v).__name__}."
                            )

                        fv = float(v)
                        if not math.isfinite(fv):
                            continue

                        batch_metric_sums[k] += fv * bs
                        batch_metric_weight_sums[k] += bs

                # dataset-level evaluator: cache only required tensors
                if dataset_required_keys:
                    for key in dataset_required_keys:
                        val = self.dataset_evaluator.resolve(eval_in, key)
                        if not torch.is_tensor(val):
                            raise TypeError(
                                f"dataset_evaluator required key {key!r} resolved to {type(val).__name__}, expected Tensor. "
                                "Fix your dataset/collate/evaluator keys."
                            )
                        _append_cached(dataset_cache, key, val)

                # OOF caching for calibrators (always cache; commit only if epoch becomes best)
                if active_calib_tasks:
                    # model_out is <task>.<key_returned_by_head>; we assume logits exists.
                    for task in active_calib_tasks:
                        if task not in model_out:
                            continue
                        node = model_out[task]
                        if not isinstance(node, dict):
                            continue
                        logits = node.get("logits", None)
                        if not torch.is_tensor(logits):
                            raise KeyError(
                                f"Expected model_out[{task!r}]['logits'] Tensor for calibrator OOF logging, got {type(logits).__name__}."
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

        # aggregate epoch-level results ---
        epoch_log: dict[str, Any] = {
            "epoch": epoch,
            "val_loss": total_loss / num_batches,
        }

        # aggregate batch metrics (skip None/NaN; if nothing valid epoch-wide -> None)
        if getattr(self, "batch_evaluator", None) is not None:
            all_keys = set(batch_metric_sums.keys()) | set(batch_metric_weight_sums.keys())
            for k in sorted(all_keys):
                w = float(batch_metric_weight_sums.get(k, 0.0))
                if w <= 0.0:
                    epoch_log[f"val_batch/{k}"] = None
                else:
                    epoch_log[f"val_batch/{k}"] = batch_metric_sums.get(k, 0.0) / w

        # dataset evaluator: build epoch_inputs dict from cached tensors and run once
        dataset_primary_value: Any = None
        if getattr(self, "dataset_evaluator", None) is not None:
            epoch_inputs: dict[str, Any] = {}
            for key, ts in dataset_cache.items():
                epoch_tensor = _cat_cached_list(ts, key=key)
                _set_by_path(epoch_inputs, key, epoch_tensor)

            dm = self.dataset_evaluator(inputs=epoch_inputs)
            if not isinstance(dm, dict):
                raise TypeError(f"dataset_evaluator must return dict[str, Any], got {type(dm).__name__}.")

            for k, v in dm.items():
                epoch_log[f"val/{k}"] = v

            dataset_primary_value = dm.get(self.dataset_evaluator.primary_metric, None)

        # -------------------------
        # Best-epoch tracking + OOF commit
        # -------------------------

        # determine selection score (always maximize internally)
        if getattr(self, "dataset_evaluator", None) is not None and _is_finite_number(dataset_primary_value):
            raw_primary = float(dataset_primary_value)
            if self.dataset_evaluator.direction == "maximize":
                score = raw_primary
            else:
                score = -raw_primary
            best_raw_for_state = raw_primary
            best_metric_kind = "evaluator_primary"
        else:
            # fallback to val_loss (minimize)
            raw_loss = float(epoch_log["val_loss"])
            score = -raw_loss
            best_raw_for_state = raw_loss
            best_metric_kind = "val_loss"

        # early stopping threshold (applies to score in max sense)
        thr = self.config.early_stopping_threshold
        thr = 0.0 if thr is None else float(thr)

        # store best score in state (not part of dataclass; ok)
        best_score_prev = getattr(self.state, "_best_score", None)
        improved = (best_score_prev is None) or (score > float(best_score_prev) + thr)

        if improved:
            self.state.best_epoch = int(epoch)
            self.state.best_metric = float(best_raw_for_state)
            self.state.best_state_dict_cpu = self._get_model_state_dict_cpu()
            setattr(self.state, "_best_score", float(score))
            setattr(self.state, "_best_metric_kind", best_metric_kind)
            self.state.epochs_since_improvement = 0

            # commit OOF logits/targets for calibrators (only active + cached)
            if active_calib_tasks:
                if not hasattr(self.state, "oof_logits"):
                    self.state.oof_logits = {}
                if not hasattr(self.state, "oof_targets"):
                    self.state.oof_targets = {}

                for task in active_calib_tasks:
                    ls = oof_logits_cache.get(task, [])
                    ts = oof_targets_cache.get(task, [])
                    if len(ls) == 0 or len(ts) == 0:
                        # no samples for this task in this val epoch => store stable keys as empty
                        self.state.oof_logits[task] = torch.empty(0)
                        self.state.oof_targets[task] = torch.empty(0)
                        continue

                    # cat along batch dim
                    log_cat = torch.cat(ls, dim=0) if ls[0].ndim >= 1 else torch.stack(ls, dim=0)
                    tgt_cat = torch.cat(ts, dim=0) if ts[0].ndim >= 1 else torch.stack(ts, dim=0)

                    self.state.oof_logits[task] = log_cat
                    self.state.oof_targets[task] = tgt_cat
        else:
            self.state.epochs_since_improvement += 1

        # expose selection score if you want (useful for optuna/pruning/debug)
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
        """
        Atomic fit:
        - Optionally reset state/weights.
        - Train for up to max_epochs.
        - If val_loader is provided: validate each epoch, track best epoch, store OOF logits/targets
        only when a new best is achieved (see _validate_one_epoch).
        - Early stopping applies only when val_loader is provided.
        - Optuna reporting/pruning uses the selection score (always "maximize" internally).
        """

        # per-fit overrides (saved + restored)
        old_grad_clip = self.config.grad_clip_norm
        old_pat = self.config.early_stopping_patience
        old_thr = self.config.early_stopping_threshold
        old_rep = self.config.optuna_report_interval

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

            # seed (atomic run)
            if self.config.random_seed is not None:
                s = int(self.config.random_seed)
                torch.manual_seed(s)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(s)

            # device
            device = self.device
            if not isinstance(device, torch.device):
                device = torch.device(device)
            self.model.to(device)

            # max epochs
            run_max_epochs = int(max_epochs) if max_epochs is not None else int(self.config.max_epochs)
            if run_max_epochs <= 0:
                raise ValueError(f"max_epochs must be > 0, got {run_max_epochs}.")

            # early stopping
            patience = self.config.early_stopping_patience
            patience = None if patience is None else int(patience)
            if patience is not None and patience < 0:
                raise ValueError(f"early_stopping_patience must be >=0 or None, got {patience}.")

            report_every = int(self.config.optuna_report_interval) if self.config.optuna_report_interval is not None else 1
            if report_every <= 0:
                report_every = 1

            self._fit_called_at_least_once = True

            for ep in range(1, run_max_epochs + 1):
                self.state.epoch = ep

                train_log = self._train_one_epoch(train_loader, epoch=ep)

                # scheduler step (if without metric)

                if self.scheduler is not None and not _scheduler_expects_metric(self.scheduler):
                    self.scheduler.step()


                if val_loader is not None:
                    val_log = self._validate_one_epoch(val_loader, epoch=ep)

                    # scheduler metric-aware step (best-effort)
                    if self.scheduler is not None and _scheduler_expects_metric(self.scheduler):
                        self.scheduler.step(val_log["val_loss"])

                    # optuna report/prune
                    if trial is not None and (ep % report_every == 0):
                        score = val_log.get("__selection_score__", None)
                        if score is None:
                            # fallback: maximize -val_loss
                            score = -float(val_log["val_loss"])
                        self.maybe_report_to_trial(trial, value=float(score), step=ep)

                    # early stopping
                    if patience is not None:
                        if self.state.epochs_since_improvement >= patience:
                            break

                else:
                    # no validation => still allow optuna to observe train loss if wanted
                    if trial is not None and (ep % report_every == 0):
                        # maximize negative train loss
                        score = -float(train_log["train_loss"])
                        self.maybe_report_to_trial(trial, value=float(score), step=ep)

            # store run summary in history if you want
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

            return self

        finally:
            # restore config overrides
            self.config.grad_clip_norm = old_grad_clip
            self.config.early_stopping_patience = old_pat
            self.config.early_stopping_threshold = old_thr
            self.config.optuna_report_interval = old_rep

    # -------------------------
    # Optuna helper hooks (optional usage in your future fit implementation)
    # -------------------------

    def maybe_report_to_trial(
        self,
        trial: Optional[optuna.trial.Trial],
        *,
        value: float,
        step: int,
    ) -> None:
        """Call trial.report and potentially raise TrialPruned."""
        if trial is None:
            return
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # -------------------------
    # Internals
    # -------------------------


    def _get_model_state_dict_cpu(self) -> dict[str, torch.Tensor]:
        # graph-disconnected cpu-based copy of model's state_dict (for stable snapshotting)
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def _load_initial_state_dict_cpu(
        self,
        *,
        model: TorchkitModel,
        model_initial_state_path: Optional[str],
    ) -> dict[str, torch.Tensor]:
        if model_initial_state_path is None:
            # graph-disconnected cpu-based copy of model's state_dict at init time (reset point)
            return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        # path was given => load from disk to CPU/RAM
        sd = torch.load(model_initial_state_path, map_location="cpu")
        if not isinstance(sd, dict):
            raise TypeError("Loaded initial state is not a state_dict (expected dict[str, Tensor]).")
        return sd

    def _restore_model_weights_from_cpu_snapshot(self) -> None:
        # move snapshot tensors to model device
        device = torch.device(self.device) if not isinstance(self.device, torch.device) else self.device
        sd = {k: v.to(self.device, non_blocking=True) for k, v in self._initial_state_dict_cpu.items()}
        self.model.load_state_dict(sd, strict=True)

    def _rebuild_stateful_objects_from_config(
        self,
        *,
        rebuild_optimizer: bool = True,
        rebuild_scheduler: bool = True,
        rebuild_scaler: bool = True,
    ) -> None:
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


# NOTE: Dataset __collate_fn__ should:
# 1. always return stable keys
# 2. missing masks should be encoded as masks with None values
# 3. [batch]["x"] must exist and be a tensor
# 4. Targets from batch should be under batch[task][y/target/targets/label/labels]
#    or at least [batch][y/target/targets/label/labels] for single-task.
# 5. [batch]["x"] and [batch]["y/target/targets/label/labels"] should have the same batch size (dim=0)
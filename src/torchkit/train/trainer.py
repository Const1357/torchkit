from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Literal
import copy

import torch
from torch import nn, Tensor

import optuna

from torchkit.models._interface import TorchkitModel
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

    scheduler_cls: Optional[type[torch.optim.lr_scheduler._LRScheduler]] = None
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
    global_step: int = 0

    best_metric: Optional[float] = None # evaluator's primary, or val loss if no evaluator
    best_epoch: Optional[int] = None

    # could store early stopping counters etc.
    epochs_since_improvement: int = 0

    # simple per-epoch logs (optional; you can remove)
    train_logs: list[dict[str, Any]] = field(default_factory=list)
    val_logs: list[dict[str, Any]] = field(default_factory=list)


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
        config: Optional[TrainerConfig] = None,
        dataset_evaluator: Optional[Evaluator] = None,
        batch_evaluator: Optional[Evaluator] = None,
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

        Behavior:
        - Computes mean val loss over batches (like train).
        - If `self.batch_evaluator` is set: runs it per batch and aggregates **scalar numeric** metrics
        as a sample-weighted mean over the epoch.
        - If `self.dataset_evaluator` is set: caches (detached, CPU) tensors for its required_keys across
        all batches, concatenates them at epoch end, and runs it once to produce dataset-level metrics.

        Notes / assumptions:
        - `batch` is dict[str, Any] and contains "x" (enforced convention).
        - Evaluator.required_keys are "/"-paths that can be resolved from a nested dict `eval_in`
        where we insert `eval_in["batch"] = batch`.
        - Dataset-level evaluator required keys must resolve to Tensors whose first dim is the sample dim.
        """

        import numbers
        from collections import defaultdict

        self.model.eval()

        device = self.device
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # helpers ---
        def _infer_batch_size(batch_dict: dict[str, Any]) -> int:
            x = batch_dict.get("x", None)
            if torch.is_tensor(x):
                if x.ndim == 0:
                    raise ValueError("batch['x'] is scalar; cannot infer batch size.")
                return int(x.shape[0])
            if isinstance(x, dict):
                # find first tensor leaf
                for v in x.values():
                    if torch.is_tensor(v):
                        if v.ndim == 0:
                            continue
                        return int(v.shape[0])
            # fallback: look anywhere in batch
            for v in batch_dict.values():
                if torch.is_tensor(v) and v.ndim >= 1:
                    return int(v.shape[0])
            raise ValueError("Could not infer batch size from batch; expected at least one Tensor with ndim>=1.")

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
            # detach+cpu to avoid holding graph / GPU memory
            cache[key].append(tensor.detach().cpu())

        def _cat_cached_list(ts: list[torch.Tensor], key: str) -> torch.Tensor:
            if len(ts) == 0:
                raise ValueError(f"Empty cache list for key {key!r} (unexpected).")
            if len(ts) == 1:
                return ts[0]
            # handle scalar tensors: stack
            if ts[0].ndim == 0:
                return torch.stack(ts, dim=0)
            # otherwise concatenate along sample dim
            return torch.cat(ts, dim=0)


        # setup caches / accumulators ---

        dataset_required_keys: tuple[str, ...] = ()
        if getattr(self, "dataset_evaluator", None) is not None:
            dataset_required_keys = tuple(self.dataset_evaluator.required_keys)  # union already for CompositeEvaluator
        dataset_cache: dict[str, list[torch.Tensor]] = defaultdict(list)

        batch_metric_sums: dict[str, float] = defaultdict(float)
        batch_metric_weight_sums: dict[str, float] = defaultdict(float)

        total_loss = 0.0
        num_batches = 0

        # validation loop ---
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

                # loss return type (scalar tensor) is validated inside objective.

                total_loss += float(loss.detach().item())
                num_batches += 1

                # batch-level evaluator (optional): scalar numeric metrics, allow None/NaN skipping
                if getattr(self, "batch_evaluator", None) is not None:
                    bm = self.batch_evaluator(inputs=eval_in)
                    if not isinstance(bm, dict):
                        raise TypeError(f"batch_evaluator must return dict[str, Any], got {type(bm).__name__}.")

                    # Track per-metric weight separately (since some metrics may be None/NaN per batch)
                    for k, v in bm.items():
                        if v is None:
                            continue  # skip missing metrics (e.g., mask absent)
                        if isinstance(v, bool):
                            raise TypeError(f"batch_evaluator metric {k!r} is bool; expected a numeric scalar or None.")
                        if not isinstance(v, numbers.Number):
                            raise TypeError(
                                f"batch_evaluator metric {k!r} must be a python number (float/int) or None for aggregation, "
                                f"got {type(v).__name__}."
                            )

                        fv = float(v)
                        # skip NaN / inf
                        if not (fv == fv) or fv == float("inf") or fv == float("-inf"):
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

        if num_batches == 0:
            raise ValueError("val_loader produced 0 batches.")

        # aggregate epoch-level results ---
        epoch_log: dict[str, Any] = {
            "epoch": epoch,
            "val_loss": total_loss / num_batches,
        }

        # aggregate batch metrics (skip None/NaN; if nothing valid epoch-wide -> None)
        if getattr(self, "batch_evaluator", None) is not None:
            # Do NOT require any global weight now; metrics may be missing for the whole epoch.
            # We aggregate per-metric with its own valid-weight sum.
            # Include keys that appeared either in sums or weights (defensive).
            all_keys = set(batch_metric_sums.keys()) | set(batch_metric_weight_sums.keys())
            for k in sorted(all_keys):
                w = float(batch_metric_weight_sums.get(k, 0.0))
                if w <= 0.0:
                    epoch_log[f"val_batch/{k}"] = None
                else:
                    epoch_log[f"val_batch/{k}"] = batch_metric_sums.get(k, 0.0) / w

        # dataset evaluator: build epoch_inputs dict from cached tensors and run once
        if getattr(self, "dataset_evaluator", None) is not None:
            epoch_inputs: dict[str, Any] = {}
            for key, ts in dataset_cache.items():
                epoch_tensor = _cat_cached_list(ts, key=key)
                _set_by_path(epoch_inputs, key, epoch_tensor)

            dm = self.dataset_evaluator(inputs=epoch_inputs)
            if not isinstance(dm, dict):
                raise TypeError(f"dataset_evaluator must return dict[str, Any], got {type(dm).__name__}.")

            # dataset metrics can include curves / lists / dicts, so we just attach them
            for k, v in dm.items():
                epoch_log[f"val/{k}"] = v

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
        Fit for `max_epochs` using current config, with optional per-fit overrides.

        NOTE: intentionally unimplemented.
        """
        raise NotImplementedError

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
        """Call trial.report and potentially raise TrialPruned (policy is up to you)."""
        if trial is None:
            return
        trial.report(value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # -------------------------
    # Internals
    # -------------------------


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
            self._scaler = torch.amp.GradScaler("cuda") if self.config.use_amp else None

# TODO: 
# 1. implement fit().
#    - early stopping
#    - optuna reporting/pruning
#    - validation every N epochs (configurable)
# 2. handle resets
# 3. write the kfolds (internal & external)
#    they take a grid and reset Trainer's state and sets params.
#    QUESTION: SHOULD FIT DO KFOLD INSIDE? DECIDE

# NOTE: Dataset __collate_fn__ should:
# 1. always return stable keys
# 2. missing masks should be encoded as masks with None values
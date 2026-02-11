from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Any, Iterable

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from torch import nn, Tensor
import torch
import inspect

from sktorch.modules.nn.FeatureAdapters import _BaseAdapter, AdapterFactory
from sktorch.modules.nn.models._base._estimator import SKTorchEstimatorBase
from sktorch.modules.nn.models.backbones.backbone import BackboneOut
from sktorch.modules.nn.models.factory import ModuleFactory


@dataclass(frozen=True)
class MultitaskerOut:
    heads_out: Dict[str, Any]
    backbone_details: Dict[str, Any] = field(default_factory=dict)
    heads_details: Dict[str, Any] = field(default_factory=dict)


class SKTorchMultitasker(SKTorchEstimatorBase, ABC):
    """
    Multitask interface.

    Notes for backbone designers:
    - If your backbone can save compute by skipping branches, implement a kwarg:
        requested_features: set[str] | None
      and only compute/return those features (omit others from the dict).
    """

    def __init__(
        self,
        *,
        backbone_factory: ModuleFactory,
        head_factories: Mapping[str, ModuleFactory],                 # task_name -> head factory
        adapter_factories: Mapping[str, AdapterFactory] | None = None,  # task_name -> adapter factory
        backbone_feature_for_task: Mapping[str, str],                 # task_name -> backbone feature key
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if not isinstance(head_factories, Mapping) or len(head_factories) == 0:
            raise ValueError("head_factories must be a non-empty mapping: task_name -> ModuleFactory.")

        if adapter_factories is None:
            adapter_factories = {
                task_name: AdapterFactory(cls_path="sktorch.modules.nn.FeatureAdapters:IdentityAdapter")
                for task_name in head_factories.keys()
            }

        # enforce aligned keys
        if set(adapter_factories.keys()) != set(head_factories.keys()):
            raise ValueError(
                f"adapter_factories keys {set(adapter_factories.keys())} do not match head_factories keys {set(head_factories.keys())}."
            )

        if set(backbone_feature_for_task.keys()) != set(head_factories.keys()):
            raise ValueError(
                f"backbone_feature_for_task keys {set(backbone_feature_for_task.keys())} do not match head_factories keys {set(head_factories.keys())}."
            )

        # sklearn params (must be stored exactly)
        self.backbone_factory = backbone_factory
        self.head_factories = dict(head_factories)
        self.adapter_factories = dict(adapter_factories)
        self.backbone_feature_for_task = dict(backbone_feature_for_task)

        super().__init__(device=device, dtype=dtype)

        # lazy modules
        self.backbone: Optional[nn.Module] = None
        self.feature_adapters: Dict[str, _BaseAdapter] = {}
        self.heads: Dict[str, nn.Module] = {}

        # backbone capability (detected once)
        self._backbone_supports_requested_features: Optional[bool] = None

        # fast-forward cache (keyed by active task set)
        self._fast_cache: Dict[frozenset[str], tuple[tuple[str, ...], tuple[str, ...]]] = {}

        # --- fitted attrs (sklearn-like). Trainer should fill these. ---
        self.task_names_: tuple[str, ...] = tuple(self.head_factories.keys())
        self.task_info_: Dict[str, Dict[str, Any]] = {}
        self.task_fitted_: Dict[str, bool] = {t: False for t in self.task_names_}

        self.to(self._device)

    # fitted state ------------------

    def _fitted_state_keys(self) -> tuple[str, ...]:
        return super()._fitted_state_keys() + ("task_names_", "task_info_", "task_fitted_")

    def _require_tasks_fitted(self, tasks: Iterable[str]) -> None:
        missing = [t for t in tasks if not self.task_fitted_.get(t, False)]
        if missing:
            raise RuntimeError(
                f"Requested task(s) not fitted: {missing}. "
                f"task_fitted_={self.task_fitted_}"
            )

    # internal helpers ---

    def _ensure_backbone(self) -> None:
        if self.backbone is not None:
            return

        backbone = self.backbone_factory.build()
        if not isinstance(backbone, nn.Module):
            raise TypeError(
                f"Backbone {self.backbone_factory.cls_path} did not produce nn.Module, got {type(backbone)}."
            )

        self.backbone = backbone
        self.add_module("mt_backbone", backbone)
        self.to(self._device)

        # detect support for compute gating kwarg
        # Backbone designers: implement `requested_features` kwarg to allow early-exit / skipping branches.
        try:
            sig = inspect.signature(backbone.forward)  # type: ignore[attr-defined]
            self._backbone_supports_requested_features = ("requested_features" in sig.parameters)
        except (TypeError, ValueError):
            self._backbone_supports_requested_features = False

    def _ensure_adapter_for_task(self, task_name: str) -> None:
        if task_name in self.feature_adapters:
            return

        af = self.adapter_factories[task_name]
        adapter = af.build()
        if not isinstance(adapter, _BaseAdapter):
            raise TypeError(
                f"Adapter {af.cls_path} for task '{task_name}' did not produce _BaseAdapter, got {type(adapter)}."
            )

        self.feature_adapters[task_name] = adapter
        self.add_module(f"adapter__{task_name}", adapter)
        self.to(self._device)

    def _ensure_head_for_task(self, task_name: str, dummy: Tensor) -> None:
        if task_name in self.heads:
            return

        hf = self.head_factories[task_name]
        head = hf.from_input(dummy)
        if not isinstance(head, nn.Module):
            raise TypeError(
                f"Head {hf.cls_path} for task '{task_name}' did not produce nn.Module, got {type(head)}."
            )

        self.heads[task_name] = head
        self.add_module(f"head__{task_name}", head)
        self.to(self._device)

    def _normalize_active_tasks(self, active_tasks: Optional[Iterable[str]]) -> tuple[str, ...]:
        all_tasks = tuple(self.head_factories.keys())
        if active_tasks is None:
            return all_tasks

        s = set(active_tasks)
        unknown = s.difference(all_tasks)
        if unknown:
            raise KeyError(f"Unknown task(s) in active_tasks: {sorted(unknown)}. Available: {list(all_tasks)}")

        # preserve a stable order (mapping order) but only include active tasks
        return tuple(t for t in all_tasks if t in s)

    # forward ---

    @override
    def forward(
        self,
        X: Tensor,
        *,
        active_tasks: Optional[Iterable[str]] = None,
        enforce_fitted: bool = False,
        backbone_fwd_args: Dict[str, Any] | None = None,
        adapter_fwd_args: Mapping[str, Dict[str, Any]] | None = None,  # task_name -> kwargs
        head_fwd_args: Mapping[str, Dict[str, Any]] | None = None,     # task_name -> kwargs
        **kwargs: Any
    ) -> MultitaskerOut:

        backbone_fwd_args = {} if backbone_fwd_args is None else backbone_fwd_args
        adapter_fwd_args = {} if adapter_fwd_args is None else dict(adapter_fwd_args)
        head_fwd_args = {} if head_fwd_args is None else dict(head_fwd_args)

        tasks = self._normalize_active_tasks(active_tasks)
        if enforce_fitted:
            self._require_tasks_fitted(tasks)

        tasks_key = frozenset(tasks)

        # lazy backbone init
        self._ensure_backbone()
        if self.backbone is None:
            raise RuntimeError("Backbone was not initialized (unexpected).")
        if self._backbone_supports_requested_features is None:
            raise RuntimeError("Backbone capability detection failed (unexpected).")

        # compute gating: request only the feature keys needed for the active tasks
        requested_keys = {self.backbone_feature_for_task[t] for t in tasks}

        bb_kwargs = dict(backbone_fwd_args)
        if self._backbone_supports_requested_features:
            bb_kwargs["requested_features"] = requested_keys

        bb_out: BackboneOut = self.backbone(X, **bb_kwargs)

        feats = bb_out.features
        if not isinstance(feats, dict):
            raise TypeError(
                "Multitasker requires BackboneOut.features to be a dict[str, Tensor] (optionally omitting keys). "
                f"Got {type(feats)}."
            )

        # fast-forward routing cache (task order + corresponding feature keys)
        cached = self._fast_cache.get(tasks_key)
        if cached is None:
            task_order = tasks
            feature_keys = tuple(self.backbone_feature_for_task[t] for t in task_order)
            self._fast_cache[tasks_key] = (task_order, feature_keys)
        else:
            task_order, feature_keys = cached

        # route backbone features -> per-task adapter -> per-task head input
        per_task_head_input: Dict[str, Tensor] = {}
        for task_name, feature_key in zip(task_order, feature_keys):
            # We accept BOTH:
            # - missing key (preferred when feature not computed)
            # - present but None (also treated as "not computed")
            if feature_key not in feats or feats[feature_key] is None:
                raise KeyError(
                    f"Backbone did not provide feature '{feature_key}' required for active task '{task_name}'. "
                    f"Requested features: {sorted(requested_keys)}. Available: {list(feats.keys())}"
                )

            x = feats[feature_key]
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"Backbone feature '{feature_key}' for task '{task_name}' must be a Tensor, got {type(x)}."
                )
            if x.ndim < 2:
                raise ValueError(
                    f"Backbone feature '{feature_key}' for task '{task_name}' must be at least 2D "
                    f"[BatchDimension, ...], got {tuple(x.shape)}."
                )

            self._ensure_adapter_for_task(task_name)
            adapter = self.feature_adapters[task_name]

            a_kwargs = adapter_fwd_args.get(task_name, {})
            head_in = adapter(x, **a_kwargs)

            if not isinstance(head_in, torch.Tensor):
                raise TypeError(f"Adapter output for task '{task_name}' must be a Tensor, got {type(head_in)}.")
            if head_in.ndim < 2:
                raise ValueError(
                    f"Head input for task '{task_name}' must be at least 2D [BatchDimension, ...], "
                    f"got {tuple(head_in.shape)}. Check your adapter."
                )

            per_task_head_input[task_name] = head_in

        # lazy heads init per task (needs per-task dummy inputs)
        for task_name in task_order:
            self._ensure_head_for_task(task_name, per_task_head_input[task_name])

        # run heads
        heads_out: Dict[str, Any] = {}
        heads_details: Dict[str, Any] = {}
        for task_name in task_order:
            head = self.heads[task_name]
            h_kwargs = head_fwd_args.get(task_name, {})
            out = head(per_task_head_input[task_name], **h_kwargs)
            heads_out[task_name] = out

            details = getattr(out, "details", None)
            if isinstance(details, dict):
                heads_details[task_name] = details

        return MultitaskerOut(
            heads_out=heads_out,
            backbone_details=bb_out.details,
            heads_details=heads_details,
        )

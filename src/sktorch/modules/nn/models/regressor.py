from abc import ABC
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, override
from torch import nn, Tensor
import torch

from sktorch.modules.nn.FeatureAdapters import _BaseAdapter, AdapterFactory
from sktorch.modules.nn.models._base._estimator import SKTorchEstimatorBase

from sktorch.modules.nn.models.backbones.backbone import BackboneOut
from sktorch.modules.nn.models.factory import ModuleFactory


@dataclass(frozen=True)
class RegressorHeadOut:
    pred: torch.Tensor
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RegressorOut:
    pred: Tensor
    backbone_details: Dict[str, Any] = field(default_factory=dict)
    reg_details: Dict[str, Any] = field(default_factory=dict)


# regressor interface class
class SKTorchRegressor(SKTorchEstimatorBase, ABC):
    """
    Generic sklearn-compatible neural network regressor.

    This class composes three modular components:

    + Backbone → produces feature representations
    + Feature Adapter → transforms backbone features into head-ready input
    + Head → produces regression predictions

    It works out of the box for a single backbone and a single head.
    No routing, multi-branch logic, or multi-task mechanisms are implemented.

    The regressor is compatible with any backbone / adapter / head combination
    that satisfies the required interface contracts.

    ------------------------------------------------------------------

    Expected component contracts:

    Backbone:
    + Must be an `nn.Module`
    + Forward must return `BackboneOut`
      containing:
        - `features: Tensor`
        - `details: Dict[str, Any]`
    + `features` must be at least 2D:
        [BatchDimension, D, ...]

    Adapter:
    + Must derive from `_BaseAdapter`
    + Forward maps backbone features → head input
    + Output must be at least 2D:
        [BatchDimension, D, ...]

    Head:
    + Must be an `nn.Module`
    + Forward must return `RegressorHeadOut`
      containing:
        - `pred: Tensor`
        - `details: Dict[str, Any]`

    ------------------------------------------------------------------

    Initialization parameters:

    + backbone_factory (ModuleFactory):
        Factory used to lazily construct the backbone.

    + adapter_factory (AdapterFactory | None):
        Factory for the feature adapter.
        If None, defaults to IdentityAdapter.

    + head_factory (ModuleFactory):
        Factory used to lazily construct the regression head.
        Head is initialized based on the adapter output shape.

    + device / dtype:
        Passed to `SKTorchEstimatorBase`.

    ------------------------------------------------------------------

    Lazy initialization:

    The backbone, adapter, and head are constructed lazily on first forward pass.
    This allows the head to infer its input dimensionality from real data.

    ------------------------------------------------------------------

    Forward behavior:

        forward(X) -> RegressorOut

    Returns a `RegressorOut` containing:
    + pred (Tensor)
    + backbone_details (Dict)
    + reg_details (Dict)

    No loss computation is performed here.
    Training should be handled externally (e.g., via `SKTorchTrainer`).

    ------------------------------------------------------------------

    Fitted attributes:

    + `is_fitted_` (managed by training logic)

    ------------------------------------------------------------------

    Invariants:

    - Features and head inputs must have batch dimension first.
    - No implicit reduction, routing, or multi-task logic is performed.
    - This class assumes a single-output regression head.
    """

    def __init__(
        self,
        *,
        backbone_factory: ModuleFactory,
        head_factory: ModuleFactory,
        adapter_factory: AdapterFactory | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):

        # sklearn params (must be stored exactly)
        self.backbone_factory = backbone_factory
        self.adapter_factory = (
            adapter_factory
            if adapter_factory is not None
            else AdapterFactory(cls_path="sktorch.modules.nn.FeatureAdapters:IdentityAdapter")
        )
        self.head_factory = head_factory

        super().__init__(device=device, dtype=dtype)

        # lazy modules
        self.backbone: Optional[nn.Module] = None
        self.feature_adapter: Optional[_BaseAdapter] = None
        self.head: Optional[nn.Module] = None

        self.to(self._device)

    # internal helpers --------------

    def _ensure_backbone(self) -> None:
        if self.backbone is not None:
            return

        backbone = self.backbone_factory.build()
        if not isinstance(backbone, nn.Module):
            raise TypeError(
                f"Backbone {self.backbone_factory.cls_path} did not produce nn.Module, got {type(backbone)}."
            )

        self.backbone = backbone
        self.add_module("reg_backbone", backbone)
        self.to(self._device)

    def _ensure_head(self, head_input: Tensor) -> None:
        if self.head is not None:
            return
        # lazy init of head based on args.

        dummy = head_input
        head = self.head_factory.from_input(dummy)

        if not isinstance(head, nn.Module):
            raise TypeError(f"Head {self.head_factory.cls_path} did not produce nn.Module, got {type(head)}.")

        self.head = head
        self.add_module("reg_head", self.head)
        self.to(self._device)

    def _ensure_adapter(self) -> None:
        if self.feature_adapter is not None:
            return

        adapter = self.adapter_factory.build()

        if not isinstance(adapter, _BaseAdapter):
            raise TypeError(f"Adapter {self.adapter_factory.cls_path} did not produce _BaseAdapter, got {type(adapter)}.")

        self.feature_adapter = adapter
        self.add_module("feature_adapter", self.feature_adapter)
        self.to(self._device)

    @override
    def forward(
        self,
        X: Tensor,
        *,
        backbone_fwd_args: Dict[str, Any] | None = None,
        head_fwd_args: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> RegressorOut:

        backbone_fwd_args = {} if backbone_fwd_args is None else backbone_fwd_args
        head_fwd_args = {} if head_fwd_args is None else head_fwd_args

        # lazy backbone init
        self._ensure_backbone()
        assert self.backbone is not None

        bb_out: BackboneOut
        bb_out = self.backbone(X, **backbone_fwd_args)
        feats = bb_out.features

        if feats.ndim < 2:
            raise ValueError(
                f"Backbone {self.backbone_factory.cls_path} output features must be at least 2D: "
                f"[BatchDimension, D, ...], got {tuple(feats.shape)}."
            )

        # lazy adapter init
        self._ensure_adapter()
        assert self.feature_adapter is not None  # if adapter factory was not provided, IdentityAdapter should be used.

        head_input: Tensor = self.feature_adapter(feats)

        if head_input.ndim < 2:
            raise ValueError(
                f"Head input must be at least 2D: [BatchDimension, D, ...], got {tuple(head_input.shape)}. "
                "Check your feature adapter."
            )

        # lazy head init
        self._ensure_head(head_input)
        assert self.head is not None

        head_out: RegressorHeadOut
        head_out = self.head(head_input, **head_fwd_args)
        pred = head_out.pred

        return RegressorOut(
            pred=pred,
            backbone_details=bb_out.details,
            reg_details=head_out.details,
        )

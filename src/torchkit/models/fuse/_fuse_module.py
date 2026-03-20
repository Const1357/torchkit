from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

try:
    from typing import override  # py3.12+
except ImportError:
    from typing_extensions import override  # py<=3.11

import torch
from torch import Tensor, nn

from torchkit.models._spec_utils import normalize_spec_kwargs, resolve_spec_kwargs


class FuseModule(nn.Module, ABC):
    """
    Fuse modules combine multiple backbone feature tensors into a single tensor
    before passing it to the FeatureAdapter.

    New contract:
        - Input:  dict[str, Tensor] (subset slice of the backbone features)
        - Output: A single Tensor suitable for downstream adaptation.

    Notes:
        - Feature names are available if a fuse module wants them.
        - Default fusers below ignore names and fuse by values().
        - If ordering matters, implement a fuser that selects keys explicitly.
    """

    @abstractmethod
    def forward(
        self,
        features: dict[str, Tensor],
        **kwargs: Any,
    ) -> Tensor:
        """Fuse multiple backbone features into a single Tensor."""
        raise NotImplementedError(
            "Subclasses of FuseModule must implement forward(features: dict[str, Tensor], **kwargs) -> Tensor."
        )

    def to_spec(self):
        from torchkit.models.fuse.factory import FuseModuleSpec

        return FuseModuleSpec(
            cls=self.__class__,
            kwargs=resolve_spec_kwargs(self),
        )


class ConcatFuseModule(FuseModule):
    """
    Channel-wise concatenation fuse.

    - Concatenates feature maps along the channel dimension (dim=1).
    - Preserves spatial structure.
    - Output shape: (B, sum(C_i), *spatial)

    Assumptions:
        - All input tensors share identical batch size and spatial dimensions.
    """

    def __init__(self, *, dim: int = 1):
        super().__init__()
        self.dim = dim
        self._spec_kwargs = normalize_spec_kwargs({"dim": dim})

    @override
    def forward(self, features: dict[str, Tensor], **kwargs: Any) -> Tensor:
        if not isinstance(features, dict) or not features:
            raise ValueError("ConcatFuseModule expects a non-empty dict[str, Tensor].")
        vals = list(features.values())
        return torch.cat(vals, dim=self.dim)

    def to_spec(self):
        return super().to_spec()


class SumFuseModule(FuseModule):
    """
    Elementwise summation fuse.

    - Stacks features and sums them elementwise.
    - Preserves spatial structure and channel dimensionality.
    - Output shape: same as each input tensor.

    Assumptions:
        - All input tensors have identical shape.
    """

    def __init__(self, *, stack_dim: int = 0):
        super().__init__()
        self.stack_dim = stack_dim
        self._spec_kwargs = normalize_spec_kwargs({"stack_dim": stack_dim})

    @override
    def forward(self, features: dict[str, Tensor], **kwargs: Any) -> Tensor:
        if not isinstance(features, dict) or not features:
            raise ValueError("SumFuseModule expects a non-empty dict[str, Tensor].")
        vals = list(features.values())
        return torch.stack(vals, dim=self.stack_dim).sum(dim=self.stack_dim)

    def to_spec(self):
        return super().to_spec()


class TabularConcatFuseModule(FuseModule):
    """
    Structural fusion: concatenate backbone feature tensor(s) with tabular tensor from payload.

    ### *Contract*:
      - features: dict[str, Tensor] (subset slice of required backbone features)
      - payload[tabular_key]: Tensor of shape (B, T) OR already broadcastable

    ### *Behavior*:
      - If multiple feature tensors are provided, they are concatenated first along `dim`.
      - If fused feature tensor is spatial (ndim > 2) and tabular is (B, T),
        tabular is reshaped to (B, T, 1, 1, ...) and expanded across spatial dims.
      - Returns torch.cat([features_fused, tabular_broadcasted], dim=dim)

    ### *Note*
      - No learnable projections here by design. Use FeatureAdapter for shaping/projection.
    """

    def __init__(
        self,
        *,
        tabular_key: str = "tabular",
        dim: int = 1,
    ):
        super().__init__()
        if not isinstance(tabular_key, str) or not tabular_key.strip():
            raise TypeError("`tabular_key` must be a non-empty str.")
        self.tabular_key = tabular_key
        self.dim = dim
        self._spec_kwargs = normalize_spec_kwargs(
            {
                "tabular_key": tabular_key,
                "dim": dim,
            }
        )

    @override
    def forward(
        self,
        features: dict[str, Tensor],
        *,
        payload: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tensor:
        if not isinstance(features, dict) or not features:
            raise ValueError("TabularConcatFuseModule expects a non-empty dict[str, Tensor].")

        if payload is None or self.tabular_key not in payload:
            raise KeyError(f"`payload` must contain key {self.tabular_key!r} for tabular fusion.")

        tab = payload[self.tabular_key]
        if not torch.is_tensor(tab):
            raise TypeError(f"`payload[{self.tabular_key!r}]` must be a Tensor, got {type(tab).__name__}.")

        feat_vals = list(features.values())
        x = feat_vals[0] if len(feat_vals) == 1 else torch.cat(feat_vals, dim=self.dim)

        # Normalize tabular to be concat-compatible with x
        # Common case: tab is (B, T)
        if tab.ndim == 2 and x.ndim > 2:
            view_shape = (tab.shape[0], tab.shape[1]) + (1,) * (x.ndim - 2)
            tab = tab.view(view_shape).expand((tab.shape[0], tab.shape[1]) + x.shape[2:])

        # If x is (B, C) and tab is (B, T, 1, 1, ...) or other mismatch, fail loudly
        if tab.ndim != x.ndim:
            raise ValueError(
                f"Tabular tensor ndim must match fused feature tensor ndim after broadcasting. "
                f"Got tab.ndim={tab.ndim}, x.ndim={x.ndim}. "
                f"(tab shape={tuple(tab.shape)}, x shape={tuple(x.shape)})"
            )

        if tab.shape[0] != x.shape[0]:
            raise ValueError(
                f"Batch size mismatch between tabular and features: "
                f"tab.shape[0]={tab.shape[0]} vs x.shape[0]={x.shape[0]}."
            )

        return torch.cat([x, tab], dim=self.dim)

    def to_spec(self):
        return super().to_spec()

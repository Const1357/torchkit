"""
Feature adapters transform raw backbone features into a representation suitable for the classifier head.
They primarily control the bias–variance tradeoff by deciding how much spatial structure is preserved
or collapsed before classification.

Below are the implemented feature adapters, ordered from **lowest variance / strongest inductive bias**
to **highest variance / most expressive**, along with their risks and recommended use cases.

---

**IdentityAdapter**
- Passes backbone features through unchanged; no spatial reduction or inductive bias.
- High reliance on the head; unsafe unless the head is explicitly designed for spatial inputs.
- Recommended use: Only when the head is spatial (e.g., conv head or attention head); not suitable for MLP heads.

---

**GAPAdapter (Global Average Pooling)**
- Collapses spatial dimensions by averaging, yielding a compact channel-wise representation.
- Very low variance; may underfit tasks requiring spatial selectivity.
- Recommended use: Small datasets, weak labels, or when strong spatial invariance is desired.

---

**GMPAdapter (Global Max Pooling)**
- Selects the strongest activation per channel across space.
- Strong inductive bias toward presence detection; sensitive to noise and outliers.
- Recommended use: Detection-style classification where feature presence matters more than extent.

---

**GAPGMPConcatAdapter**
- Concatenates global average and max pooled features.
- Slightly higher variance than GAP/GMP alone; still robust and low-risk.
- Recommended use: General-purpose default when both feature strength and coverage are informative.

---

**StatsPoolAdapter**
- Aggregates spatial mean and standard deviation per channel.
- Captures spatial heterogeneity with modest increase in variance.
- Recommended use: Tasks where variability and dispersion of activations carry semantic meaning.

---

**LogSumExpPoolAdapter**
- Applies log-sum-exp pooling as a smooth approximation to max pooling.
- Moderate variance, especially if temperature is learnable; sensitive to hyperparameters.
- Recommended use: Intermediate regime between GAP and GMP when soft selection is beneficial.

---

**GeMAdapter (Generalized Mean Pooling)**
- Interpolates between average and max pooling via a (optionally learnable) exponent.
- Increased flexibility; may overfit on small datasets if the exponent is learned.
- Recommended use: Medium-to-large datasets where pooling behavior should be learned rather than fixed.

---

**Conv1x1GAPAdapter**
- Projects channels with a learnable 1×1×… convolution before global average pooling.
- Higher capacity; prone to overfitting if channel dimensionality is large.
- Recommended use: When channel reweighting or compression is required and sufficient data is available.

---

**AttnPoolAdapter**
- Learns spatial attention weights to compute a weighted feature summary.
- Expressive but data-hungry; unstable on small or noisy datasets.
- Recommended use: Large datasets where spatial focus varies across samples and interpretability is desired.

---

**TokenAdapter**
- Uses multiple learned token queries to attend over spatial dimensions.
- High variance and parameter count; requires strong regularization and sufficient data.
- Recommended use: Large-scale settings where multiple spatial concepts must be jointly modeled.

---

**SPPAdapter (Spatial Pyramid Pooling)**
- Pools features at multiple spatial resolutions and concatenates results.
- Produces very high-dimensional representations; downstream head often overfits.
- Recommended use: When multi-scale spatial context is essential and head capacity is tightly controlled.

---

**FlattenAdapter**
- Flattens all spatial dimensions into a single vector.
- No spatial inductive bias and extreme dimensionality; almost always overfits in practice.
- Recommended use: Almost never; only for controlled experiments or very large datasets with heavy regularization.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Dict, Sequence, override, Type, Any
from torch import (Tensor, nn)
import torch

from sktorch.modules.nn._util import _as_device, _cls_to_path, _import_from_path



class _BaseAdapter(nn.Module, ABC):
    """Base class for Feature Adapters."""
    @abstractmethod
    def forward(self, features: Tensor) -> Tensor:
        """Transforms backbone features into a format suitable for the head."""
        raise NotImplementedError("Subclasses of _BaseAdapter must implement the forward method to transform features.")



@dataclass
class AdapterFactory:
    cls_path: str
    kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_type(cls, t: Type[Any], **kwargs: Any) -> "AdapterFactory":
        return cls(cls_path=_cls_to_path(t), kwargs=kwargs)

    def build(self) -> _BaseAdapter:
        """Builds the adapter instance."""
        cls = _import_from_path(self.cls_path)
        adapter = cls(**self.kwargs)
        if not isinstance(adapter, _BaseAdapter):
            raise TypeError(f"Factory {self.cls_path}: Built object is not a _BaseAdapter, got {type(adapter)}.")
        return adapter

# utilities
def _spatial_dims(x: Tensor) -> tuple[int, ...]:
    if x.ndim < 3:
        raise ValueError(
            f"Expected features with shape (B, C, *spatial) and ndim>=3, got shape={tuple(x.shape)}."
        )
    return tuple(range(2, x.ndim))

# helpers to support ND adaptive pooling without branching everywhere
# PyTorch only has adaptive_{avg,max}_pool{1,2,3}d. We provide tiny ND dispatch.
def _adaptive_avg_pool(features: Tensor, target: Sequence[int]) -> Tensor:
    nd = len(target)
    if nd == 1:
        return torch.nn.functional.adaptive_avg_pool1d(features, target[0])
    if nd == 2:
        return torch.nn.functional.adaptive_avg_pool2d(features, target)
    if nd == 3:
        return torch.nn.functional.adaptive_avg_pool3d(features, target)
    raise ValueError(f"Adaptive pooling not implemented for nd={nd}. For nd>3, use GAP/GeM/Attn, or downsample in backbone.")


def _adaptive_max_pool(features: Tensor, target: Sequence[int]) -> Tensor:
    nd = len(target)
    if nd == 1:
        return torch.nn.functional.adaptive_max_pool1d(features, target[0])
    if nd == 2:
        return torch.nn.functional.adaptive_max_pool2d(features, target)
    if nd == 3:
        return torch.nn.functional.adaptive_max_pool3d(features, target)
    raise ValueError(f"Adaptive pooling not implemented for nd={nd}. For nd>3, use GAP/GeM/Attn, or downsample in backbone.")



def _spp_forward(self: SPPAdapter, features: Tensor) -> Tensor:
    dims = _spatial_dims(features)
    nd = len(dims)

    outs = []
    for b in self.bins:
        target = [b] * nd
        pooled = _adaptive_avg_pool(features, target) if self.mode == "avg" else _adaptive_max_pool(features, target)
        outs.append(pooled.reshape(features.size(0), -1))

    return torch.cat(outs, dim=1)
SPPAdapter.forward = _spp_forward  # type: ignore[method-assign]



class IdentityAdapter(_BaseAdapter):

    @override
    def forward(self, features: Tensor) -> Tensor:
        """Passes features through unchanged."""
        return features
    
class FlattenAdapter(_BaseAdapter):

    @override
    def forward(self, features: Tensor) -> Tensor:
        """Flattens features into a 2D tensor (batch_size, -1)."""
        return features.view(features.size(0), -1)
    

class GAPAdapter(_BaseAdapter):
    """
    Global Average Pooling over all spatial dimensions.
    Assumes shape: (B, C, spatial...), outputs: (B, C) by default.
    """

    def __init__(self, keepdim: bool = False) -> None:
        super().__init__()
        self.keepdim = keepdim

    @override
    def forward(self, features: Tensor) -> Tensor:
        dims = _spatial_dims(features)
        return features.mean(dim=dims, keepdim=self.keepdim)
    
class GMPAdapter(_BaseAdapter):
    """
    Global Max Pooling over all spatial dimensions.
    Assumes shape: (B, C, spatial...), outputs: (B, C) by default.
    """

    def __init__(self, keepdim: bool = False) -> None:
        super().__init__()
        self.keepdim = keepdim

    @override
    def forward(self, features: Tensor) -> Tensor:
        dims = _spatial_dims(features)
        return features.amax(dim=dims, keepdim=self.keepdim)
    
class GAPGMPConcatAdapter(_BaseAdapter):
    """
    Concatenates GAP and GMP: out = [mean, max] along channel dim.
    Input:  (B, C, spatial...)
    Output: (B, 2C)
    """

    @override
    def forward(self, features: Tensor) -> Tensor:
        dims = _spatial_dims(features)
        gap = features.mean(dim=dims)
        gmp = features.amax(dim=dims)
        return torch.cat([gap, gmp], dim=1)
    

class StatsPoolAdapter(_BaseAdapter):
    """
    Concatenates mean and std over spatial dims.
    Input:  (B, C, spatial...)
    Output: (B, 2C)

    Note: std uses unbiased=False for stability on small spatial grids.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    @override
    def forward(self, features: Tensor) -> Tensor:
        dims = _spatial_dims(features)
        mean = features.mean(dim=dims)
        # var = E[x^2] - (E[x])^2 is typically stable and ND-friendly
        mean2 = (features * features).mean(dim=dims)
        var = (mean2 - mean * mean).clamp_min(0.0)
        std = (var + self.eps).sqrt()
        return torch.cat([mean, std], dim=1)
    

class GeMAdapter(_BaseAdapter):
    """
    Generalized Mean Pooling (GeM).
      GeM(x) = (mean(x^p))^(1/p)

    **If your features can be negative, consider applying an activation (>0) before this adapter.**

    - p can be learnable (default).
    - For numerical stability, clamps inputs to eps.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable_p: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        if learnable_p:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))

    @override
    def forward(self, features: Tensor) -> Tensor:
        dims = _spatial_dims(features)
        p = self.p.clamp_min(1.0)  # avoid pathological values
        x = features.clamp_min(self.eps)
        return x.pow(p).mean(dim=dims).pow(1.0 / p)
    

class LogSumExpPoolAdapter(_BaseAdapter):
    """
    LogSumExp pooling (smooth max) over spatial dims:
      y = (1/t) * log(mean(exp(t*x)))  (up to constants)

    Temperature t:
    - higher => closer to max
    - lower => closer to mean (linear region)
    """

    def __init__(self, temperature: float = 1.0, learnable: bool = False, eps: float = 1e-12) -> None:
        super().__init__()
        self.eps = float(eps)
        if learnable:
            self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        else:
            self.register_buffer("temperature", torch.tensor(float(temperature)))

    @override
    def forward(self, features: Tensor) -> Tensor:
        dims = _spatial_dims(features)
        t = self.temperature.clamp_min(self.eps)

        # numerically stable LSE: log(mean(exp(t*x))) = logsumexp(t*x) - log(S)
        x = features * t
        lse = torch.logsumexp(x, dim=dims)  # (B, C)
        # compute log(S)
        S = 1
        for d in features.shape[2:]:
            S *= int(d)
        lse = lse - torch.log(torch.tensor(float(S), device=features.device, dtype=features.dtype))
        return lse / t
    


class AttnPoolAdapter(_BaseAdapter):
    """
    Attention pooling: learns a spatial weighting and returns weighted sum.

    Mechanics (channel-first):
    - flatten spatial dims to S
    - produce attention scores a over S (per-sample), softmax over S
    - output y = Σ_s attn_s * x_s  -> (B, C)

    score_mode:
    - "shared": one score per location (uses a 1x1 conv over all channels after flattening)
    - "per_channel": per-channel scores then normalize (heavier; often unnecessary)
    """

    def __init__(self, in_channels: int, score_mode: str = "shared") -> None:
        super().__init__()
        if score_mode not in ("shared", "per_channel"):
            raise ValueError(f"score_mode must be 'shared' or 'per_channel', got {score_mode!r}")
        self.score_mode = score_mode

        if score_mode == "shared":
            self.score = nn.Conv1d(in_channels, 1, kernel_size=1, bias=True)
        else:
            self.score = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)

    @override
    def forward(self, features: Tensor) -> Tensor:
        # dims = _spatial_dims(features)
        B, C = features.shape[:2]
        S = 1
        for d in features.shape[2:]:
            S *= int(d)

        xf = features.reshape(B, C, S)  # (B, C, S)

        a = self.score(xf).softmax(dim=-1)  # (B, C, S)
        y = (xf * a).sum(dim=-1)            # (B, C)

        return y
    

class SPPAdapter(_BaseAdapter):
    """
    Spatial Pyramid Pooling via adaptive average pooling at multiple bin sizes.

    bins: e.g. (1,), (1,2,4) for 2D; for 3D you'd still pass (1,2,4) and it will
          create grids (b,b,b) across spatial dims.

    Output: (B, C * sum(prod([b]*ndims)) for b in bins)
    """

    def __init__(self, bins: Sequence[int] = (1, 2, 4), mode: str = "avg") -> None:
        super().__init__()
        if mode not in ("avg", "max"):
            raise ValueError(f"mode must be 'avg' or 'max', got {mode!r}")
        if len(bins) == 0:
            raise ValueError("bins must be non-empty.")
        if any(int(b) <= 0 for b in bins):
            raise ValueError(f"All bins must be positive integers, got {bins}.")
        self.bins = tuple(int(b) for b in bins)
        self.mode = mode

    @override
    def forward(self, features: Tensor) -> Tensor:
        dims = _spatial_dims(features)
        nd = len(dims)

        outs = []
        for b in self.bins:
            target = [b] * nd
            if self.mode == "avg":
                pooled = torch.nn.functional.adaptive_avg_pool_nd(features, target)  # type: ignore[attr-defined]
            else:
                pooled = torch.nn.functional.adaptive_max_pool_nd(features, target)  # type: ignore[attr-defined]
            outs.append(pooled.reshape(features.size(0), -1))

        return torch.cat(outs, dim=1)
    
# Patch SPPAdapter to use our ND dispatch without relying on non-existent F.adaptive_*_pool_nd
SPPAdapter.forward.__annotations__ = {"features": Tensor, "return": Tensor}



class Conv1x1GAPAdapter(_BaseAdapter):
    """
    Learnable channel projection using a 1x1x... conv (ND via Conv1d/2d/3d) followed by GAP.

    - Good when backbone C is large but head expects smaller dim.
    - Only supports spatial ndim in {1,2,3}. For ND>3, use a Linear on GAP output.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.bias = bool(bias)

        # conv is created lazily at first forward once we know spatial ndim
        self._conv: nn.Module | None = None
        self._gap = GAPAdapter(keepdim=False)

    def _build_conv(self, spatial_ndim: int) -> nn.Module:
        if spatial_ndim == 1:
            return nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=self.bias)
        if spatial_ndim == 2:
            return nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=self.bias)
        if spatial_ndim == 3:
            return nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=self.bias)
        raise ValueError(
            f"Conv1x1GAPAdapter only supports spatial dims 1..3, got spatial_ndim={spatial_ndim}. "
            f"For ND>3: use GAPAdapter -> Linear."
        )

    @override
    def forward(self, features: Tensor) -> Tensor:
        dims = _spatial_dims(features)
        spatial_ndim = len(dims)

        if self._conv is None:
            self._conv = self._build_conv(spatial_ndim).to(device=features.device, dtype=features.dtype)

        x = self._conv(features)  # type: ignore[operator]
        return self._gap(x)
    

class TokenAdapter(_BaseAdapter):
    """
    Token adapter (learned queries over spatial dims).

    + Input : (B, C, spatial...)
    + Output: (B, K*C)  where K = num_tokens

    For K=1 this is basically attention pooling; for K>1 it returns multiple pooled tokens.
    """

    def __init__(self, in_channels: int, num_tokens: int = 4):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive (> 0), got {num_tokens}.")
        self.in_channels = int(in_channels)
        self.num_tokens = int(num_tokens)

        # learned queries: (K, C)
        self.queries = nn.Parameter(torch.empty(self.num_tokens, self.in_channels))
        nn.init.trunc_normal_(self.queries, std=0.02)

    @override
    def forward(self, features: Tensor) -> Tensor:
        if features.ndim < 3:
            raise ValueError(
                f"TokenAdapter requires (B, C, *spatial) with ndim>=3, got {tuple(features.shape)}."
            )

        B, C = features.shape[:2]
        if C != self.in_channels:
            raise ValueError(f"TokenAdapter expected C={self.in_channels}, got C={C}.")

        # (B, C, *spatial) -> (B, S, C)
        S = 1
        for d in features.shape[2:]:
            S *= int(d)
        x = features.reshape(B, C, S).transpose(1, 2).contiguous()  # (B, S, C)

        # attention scores: (B, K, S)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)             # (B, K, C)
        scores = torch.einsum("bkc,bsc->bks", q, x) / math.sqrt(C)

        attn = torch.softmax(scores, dim=-1)                        # (B, K, S)

        # tokens: (B, K, C)
        tokens = torch.einsum("bks,bsc->bkc", attn, x)  # weighted sum over S

        # flatten: (B, K*C)
        return tokens.reshape(B, -1)
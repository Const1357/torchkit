# tests/test_sktorch_regressor.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pytest
import torch
from torch import nn, Tensor

from sktorch.modules.nn.models.regressor import SKTorchRegressor
from sktorch.modules.nn.models.factory import ModuleFactory


# -----------------------
# Dummy backbone / head
# -----------------------

@dataclass(frozen=True)
class BackboneOut:
    features: Tensor
    details: Dict[str, Any] = field(default_factory=dict)


class DummyBackbone(nn.Module):
    """
    Returns features with the same shape as input (expects >=2D).
    """
    def forward(self, x: Tensor, **kwargs: Any) -> BackboneOut:
        return BackboneOut(features=x, details={"bb_called": True, "kwargs": dict(kwargs)})


class DummyBadBackbone1D(nn.Module):
    """
    Returns 1D features to trigger regressor invariant (ndim < 2).
    """
    def forward(self, x: Tensor, **kwargs: Any) -> BackboneOut:
        # intentionally wrong: (B,) => ndim=1
        return BackboneOut(features=torch.zeros(x.shape[0], device=x.device, dtype=x.dtype), details={})


class DummyNotABackboneReturn(nn.Module):
    """
    Returns something that doesn't have .features (will error in regressor forward).
    """
    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        return {"features": x}  # wrong type


class DummyRegressorHead(nn.Module):
    """
    Expects >=2D input, outputs pred (B, out_dim).
    """
    def __init__(self, *, input_shape: tuple[int, ...], out_dim: int = 1) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.linear = nn.Linear(int(np.prod(input_shape)), self.out_dim)

    def forward(self, x: Tensor, **kwargs: Any):
        x2 = x.view(x.shape[0], -1)
        pred = self.linear(x2)
        # return exactly the expected protocol (RegressorHeadOut-like)
        return type(
            "RegressorHeadOut",
            (),
            {"pred": pred, "details": {"head_called": True, "kwargs": dict(kwargs)}},
        )()


class DummyHeadWrongReturn(nn.Module):
    """
    Returns a Tensor directly (missing .pred), should crash in regressor forward.
    """
    def __init__(self, *, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.linear = nn.Linear(int(np.prod(input_shape)), 1)

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        return self.linear(x.view(x.shape[0], -1))


class DummyBackboneNotAModule:
    def __init__(self, **kwargs: Any):
        pass


class DummyHeadNotAModule:
    def __init__(self, *, input_shape: tuple[int, ...], **kwargs: Any):
        pass


# -----------------------
# Helpers
# -----------------------

def _make_reg(*, out_dim: int = 1) -> SKTorchRegressor:
    backbone_factory = ModuleFactory.from_type(DummyBackbone)
    head_factory = ModuleFactory.from_type(DummyRegressorHead, out_dim=out_dim)

    # Use default adapter_factory=None (IdentityAdapter) to keep test assumptions minimal.
    return SKTorchRegressor(
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        device="cpu",
        dtype=torch.float32,
    )


# -----------------------
# Tests
# -----------------------

def test_forward_lazy_inits_modules_and_returns_pred():
    m = _make_reg(out_dim=1)
    assert m.backbone is None
    assert m.feature_adapter is None
    assert m.head is None

    x = torch.randn(4, 10)
    out = m(x)

    # lazy init happened
    assert m.backbone is not None
    assert m.feature_adapter is not None
    assert m.head is not None

    assert isinstance(out.pred, torch.Tensor)
    assert out.pred.shape == (4, 1)
    assert isinstance(out.backbone_details, dict)
    assert isinstance(out.reg_details, dict)


def test_forward_passes_through_backbone_fwd_args_and_head_fwd_args():
    m = _make_reg(out_dim=2)
    x = torch.randn(2, 6)

    out = m(
        x,
        backbone_fwd_args={"alpha": 123},
        head_fwd_args={"beta": "ok"},
    )

    assert out.backbone_details.get("bb_called") is True
    assert out.backbone_details.get("kwargs", {}).get("alpha") == 123
    assert out.reg_details.get("head_called") is True
    assert out.reg_details.get("kwargs", {}).get("beta") == "ok"
    assert out.pred.shape == (2, 2)


def test_backbone_features_must_be_at_least_2d():
    backbone_factory = ModuleFactory.from_type(DummyBadBackbone1D)
    head_factory = ModuleFactory.from_type(DummyRegressorHead, out_dim=1)

    m = SKTorchRegressor(
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(3, 4)
    with pytest.raises(ValueError, match="output features must be at least 2D"):
        _ = m(x)


def test_backbone_must_build_nn_module():
    backbone_factory = ModuleFactory.from_type(DummyBackboneNotAModule)
    head_factory = ModuleFactory.from_type(DummyRegressorHead, out_dim=1)

    m = SKTorchRegressor(
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(2, 5)
    with pytest.raises(TypeError, match=r"Built object is not a torch\.nn\.Module"):
        _ = m(x)


def test_head_must_build_nn_module():
    backbone_factory = ModuleFactory.from_type(DummyBackbone)
    head_factory = ModuleFactory.from_type(DummyHeadNotAModule)

    m = SKTorchRegressor(
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(2, 5)
    with pytest.raises(TypeError, match=r"Built object is not a torch\.nn\.Module"):
        _ = m(x)


def test_head_return_must_expose_pred_attribute():
    backbone_factory = ModuleFactory.from_type(DummyBackbone)
    head_factory = ModuleFactory.from_type(DummyHeadWrongReturn)

    m = SKTorchRegressor(
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(2, 5)
    with pytest.raises(AttributeError):
        _ = m(x)

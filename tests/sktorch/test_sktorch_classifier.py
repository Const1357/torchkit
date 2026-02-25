# tests/test_sktorch_classifier.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import pytest
import torch
from torch import nn, Tensor

from sktorch.modules.nn.models.classifier import SKTorchClassifier
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
    Returns 1D features to trigger classifier invariant (ndim < 2).
    """
    def forward(self, x: Tensor, **kwargs: Any) -> BackboneOut:
        # intentionally wrong: (B,) => ndim=1
        return BackboneOut(features=torch.zeros(x.shape[0], device=x.device, dtype=x.dtype), details={})


class DummyNotABackboneReturn(nn.Module):
    """
    Returns something that doesn't have .features (will error in classifier forward).
    """
    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        return {"features": x}  # wrong type


class DummyClassifierHead(nn.Module):
    """
    Expects >=2D input, outputs logits (B, C).
    """
    def __init__(self, *, input_shape: tuple[int, ...], n_classes: int = 3) -> None:
        super().__init__()
        # input_shape is (D, ...) excluding batch; we reduce to D by flatten
        self.n_classes = int(n_classes)
        self.linear = nn.Linear(int(np.prod(input_shape)), self.n_classes)

    def forward(self, x: Tensor, **kwargs: Any):
        x2 = x.view(x.shape[0], -1)
        logits = self.linear(x2)
        # return exactly the expected protocol (ClassifierHeadOut-like)
        return type(
            "ClassifierHeadOut",
            (),
            {"logits": logits, "details": {"head_called": True, "kwargs": dict(kwargs)}},
        )()


class DummyHeadWrongReturn(nn.Module):
    """
    Returns a Tensor directly (missing .logits), should crash in classifier forward.
    """
    def __init__(self, *, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.linear = nn.Linear(int(np.prod(input_shape)), 2)

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        return self.linear(x.view(x.shape[0], -1))


# These must be module-level so ModuleFactory.from_type(...) can produce an importable cls_path.
class DummyBackboneNotAModule:
    def __init__(self, **kwargs: Any):
        pass


class DummyHeadNotAModule:
    def __init__(self, *, input_shape: tuple[int, ...], **kwargs: Any):
        pass


# -----------------------
# Helpers
# -----------------------

def _make_clf(*, return_probs: bool = False, classes: np.ndarray | None = None) -> SKTorchClassifier:
    backbone_factory = ModuleFactory.from_type(DummyBackbone)
    head_factory = ModuleFactory.from_type(DummyClassifierHead, n_classes=3)

    # Use default adapter_factory=None (IdentityAdapter) to keep test assumptions minimal.
    return SKTorchClassifier(
        classes=classes,
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        return_probs=return_probs,
        device="cpu",
        dtype=torch.float32,
    )


# -----------------------
# Tests
# -----------------------

def test_init_sets_classes_when_provided():
    classes = np.array(["a", "b", "c"], dtype=object)
    m = _make_clf(classes=classes)
    assert hasattr(m, "classes_")
    assert np.array_equal(m.classes_, classes)


def test_forward_lazy_inits_modules_and_returns_logits_only_when_return_probs_false():
    m = _make_clf(return_probs=False)
    assert m.backbone is None
    assert m.feature_adapter is None
    assert m.head is None

    x = torch.randn(4, 10)
    out = m(x)

    # lazy init happened
    assert m.backbone is not None
    assert m.feature_adapter is not None
    assert m.head is not None

    assert isinstance(out.logits, torch.Tensor)
    assert out.logits.shape == (4, 3)
    assert out.probs is None
    assert isinstance(out.backbone_details, dict)
    assert isinstance(out.clf_details, dict)


def test_forward_returns_probs_when_return_probs_true_and_probs_are_valid():
    m = _make_clf(return_probs=True)
    x = torch.randn(5, 7)
    out = m(x)

    assert out.probs is not None
    assert out.probs.shape == (5, 3)

    # softmax sanity: each row sums ~1
    row_sums = out.probs.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_forward_passes_through_backbone_fwd_args_and_head_fwd_args():
    m = _make_clf(return_probs=False)
    x = torch.randn(2, 6)

    out = m(
        x,
        backbone_fwd_args={"alpha": 123},
        head_fwd_args={"beta": "ok"},
    )

    assert out.backbone_details.get("bb_called") is True
    assert out.backbone_details.get("kwargs", {}).get("alpha") == 123
    assert out.clf_details.get("head_called") is True
    assert out.clf_details.get("kwargs", {}).get("beta") == "ok"


def test_backbone_features_must_be_at_least_2d():
    backbone_factory = ModuleFactory.from_type(DummyBadBackbone1D)
    head_factory = ModuleFactory.from_type(DummyClassifierHead, n_classes=2)

    m = SKTorchClassifier(
        classes=None,
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        return_probs=False,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(3, 4)  # dummy input
    with pytest.raises(ValueError, match="output features must be at least 2D"):
        _ = m(x)


def test_backbone_must_build_nn_module():
    backbone_factory = ModuleFactory.from_type(DummyBackboneNotAModule)
    head_factory = ModuleFactory.from_type(DummyClassifierHead, n_classes=2)

    m = SKTorchClassifier(
        classes=None,
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        return_probs=False,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(2, 5)
    with pytest.raises(TypeError, match=r"Built object is not a torch\.nn\.Module"):
        _ = m(x)


def test_head_must_build_nn_module():
    backbone_factory = ModuleFactory.from_type(DummyBackbone)
    head_factory = ModuleFactory.from_type(DummyHeadNotAModule)

    m = SKTorchClassifier(
        classes=None,
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        return_probs=False,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(2, 5)
    with pytest.raises(TypeError, match=r"Built object is not a torch\.nn\.Module"):
        _ = m(x)


def test_head_return_must_expose_logits_attribute():
    backbone_factory = ModuleFactory.from_type(DummyBackbone)
    head_factory = ModuleFactory.from_type(DummyHeadWrongReturn)

    m = SKTorchClassifier(
        classes=None,
        backbone_factory=backbone_factory,
        head_factory=head_factory,
        adapter_factory=None,
        return_probs=False,
        device="cpu",
        dtype=torch.float32,
    )

    x = torch.randn(2, 5)
    with pytest.raises(AttributeError):
        _ = m(x)


def test_fitted_state_keys_includes_classes_():
    m = _make_clf(classes=np.array([0, 1, 2]))
    keys = m._fitted_state_keys()
    assert "is_fitted_" in keys
    assert "classes_" in keys

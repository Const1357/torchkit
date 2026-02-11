# test_objectives.py
# Run: pytest -q
from __future__ import annotations

import pytest
import torch
from torch import Tensor

from sktorch.modules.nn.objectives._base import RelationalObjective, IntrinsicObjective, ContextualObjective, LossOut
from sktorch.modules.nn.objectives.composite import CompositeObjective


# -----------------------
# Minimal dummy objectives
# -----------------------

class DummyRelational(RelationalObjective):
    def __init__(self, *, required: bool = True, weight: float = 1.0):
        super().__init__(
            name="dummy_rel",
            weight=weight,
            required=required,
            required_pred_keys=("logits",),
            required_target_keys=("y",),
            required_context_keys=(),
        )

    def loss(self, predictions, targets, context=None) -> LossOut:
        logits = predictions["logits"]
        y = targets["y"]
        l = ((logits - y) ** 2).mean()  # scalar
        return LossOut(loss=l, details={"mse": l.detach()})


class DummyIntrinsic(IntrinsicObjective):
    def __init__(self, *, required: bool = True, weight: float = 1.0):
        super().__init__(
            name="dummy_int",
            weight=weight,
            required=required,
            required_pred_keys=("p",),
            required_context_keys=(),
        )

    def loss(self, predictions, context=None) -> LossOut:
        p = predictions["p"]
        l = p.mean()  # scalar
        return LossOut(loss=l, details={"mean": l.detach()})


class DummyContextual(ContextualObjective):
    def __init__(self, *, required: bool = True, weight: float = 1.0):
        super().__init__(
            name="dummy_ctx",
            weight=weight,
            required=required,
            required_context_keys=("w",),
            required_pred_keys=(),
            required_target_keys=(),
        )

    def loss(self, context, predictions=None, targets=None) -> LossOut:
        w = context["w"]
        l = (w ** 2).mean()  # scalar
        return LossOut(loss=l, details={"w2": l.detach()})


class BadNonScalarIntrinsic(IntrinsicObjective):
    def __init__(self):
        super().__init__(name="bad_nonscalar", required=True, required_pred_keys=("p",), required_context_keys=())

    def loss(self, predictions, context=None) -> LossOut:
        p = predictions["p"]
        return LossOut(loss=p, details={})  # non-scalar


# -----------------------
# Helpers
# -----------------------

def _rand(*shape, dtype=torch.float32, device=None, requires_grad: bool = False) -> Tensor:
    device = device if device is not None else torch.device("cpu")
    t = torch.randn(*shape, dtype=dtype, device=device)
    if requires_grad:
        t.requires_grad_()
    return t


# -----------------------
# Tests: base contracts
# -----------------------

def test_required_relational_missing_keys_raises_keyerror():
    obj = DummyRelational(required=True)

    with pytest.raises(KeyError):
        obj(predictions={"logits": _rand(4, 3)}, targets={})  # missing y

    with pytest.raises(KeyError):
        obj(predictions={}, targets={"y": _rand(4, 3)})  # missing logits

    with pytest.raises(KeyError):
        obj(predictions={"logits": None}, targets={"y": _rand(4, 3)})  # None counts as missing


def test_required_intrinsic_missing_keys_raises_keyerror():
    obj = DummyIntrinsic(required=True)

    with pytest.raises(KeyError):
        obj(predictions={})

    with pytest.raises(KeyError):
        obj(predictions={"p": None})


def test_required_contextual_missing_keys_raises_keyerror():
    obj = DummyContextual(required=True)

    with pytest.raises(KeyError):
        obj(context={})

    with pytest.raises(KeyError):
        obj(context={"w": None})


def test_tensor_mapping_validation_strict_types():
    obj = DummyIntrinsic(required=False)

    with pytest.raises(TypeError):
        obj(predictions={"p": "not_a_tensor"})  # type: ignore[arg-type]

    with pytest.raises(TypeError):
        obj(predictions={1: _rand(3)})  # type: ignore[dict-item]

    rel = DummyRelational(required=False)
    with pytest.raises(TypeError):
        rel(predictions={"logits": _rand(2)}, targets={"y": "nope"})  # type: ignore[arg-type]


def test_postprocess_invariants_enforced():
    class BadReturnIntrinsic(IntrinsicObjective):
        def __init__(self):
            super().__init__(name="bad", required=True, required_pred_keys=("p",), required_context_keys=())

        def loss(self, predictions, context=None):  # type: ignore[override]
            return "not_lossout"  # type: ignore[return-value]

    obj = BadReturnIntrinsic()
    with pytest.raises(TypeError):
        obj(predictions={"p": _rand(4)})

    class EmptyLossIntrinsic(IntrinsicObjective):
        def __init__(self):
            super().__init__(name="empty", required=True, required_pred_keys=("p",), required_context_keys=())

        def loss(self, predictions, context=None) -> LossOut:
            return LossOut(loss=torch.tensor([], dtype=torch.float32), details={})

    obj2 = EmptyLossIntrinsic()
    with pytest.raises(ValueError):
        obj2(predictions={"p": _rand(4)})


# -----------------------
# Tests: optional skip + zero-loss semantics
# -----------------------

def test_optional_objective_skips_and_returns_zero_when_tensor_exists_somewhere():
    """
    Optional objectives can only skip if a zero-loss can be constructed.
    Provide a tensor in context so device/dtype can be inferred.
    """
    obj = DummyIntrinsic(required=False)

    preds = {"p": None}  # missing required_pred_key
    out = obj(predictions=preds, context={"ref": _rand(1)})

    assert isinstance(out, LossOut)
    assert isinstance(out.loss, Tensor)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss).item()
    assert out.loss.item() == pytest.approx(0.0)
    assert "missing_pred_keys" in out.details


def test_optional_objective_skip_raises_if_no_tensors_anywhere():
    obj = DummyIntrinsic(required=False)
    with pytest.raises(RuntimeError):
        obj(predictions={"p": None})


def test_optional_zero_is_graph_connected_when_prediction_tensor_available():
    obj = DummyIntrinsic(required=False)

    # Missing required key, but there exists another prediction tensor to anchor on
    x = _rand(5, requires_grad=True)
    out = obj(predictions={"p": None, "other": x})

    assert out.loss.ndim == 0
    out.loss.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.zeros_like(x.grad))


def test_optional_zero_disconnected_fallback_uses_device_and_dtype_from_targets_or_context():
    rel = DummyRelational(required=False)

    # Missing pred key, but targets tensor exists to infer device/dtype for disconnected zero
    y = _rand(4, 3, dtype=torch.float64)
    out = rel(predictions={"logits": None}, targets={"y": y})

    assert out.loss.ndim == 0
    assert out.loss.device == y.device
    assert out.loss.dtype == torch.float64
    assert out.loss.item() == pytest.approx(0.0)


# -----------------------
# Tests: NaN propagation policy (basic)
# -----------------------

def test_nan_in_loss_is_not_silently_fixed():
    obj = DummyIntrinsic(required=True)
    preds = {"p": torch.tensor([float("nan"), 1.0], dtype=torch.float32)}
    out = obj(predictions=preds)

    assert torch.isnan(out.loss).item()


# -----------------------
# Tests: composite objective contracts
# -----------------------

def test_composite_weighted_sum_and_detail_keys_flattening():
    o1 = DummyIntrinsic(required=True, weight=2.0)
    o2 = DummyContextual(required=True, weight=0.5)

    comp = CompositeObjective(o1, o2, required=True)

    p = torch.tensor([1.0, 3.0], dtype=torch.float32)  # mean=2.0
    w = torch.tensor([2.0, 0.0], dtype=torch.float32)  # mean(w^2)=2.0

    out = comp(predictions={"p": p}, context={"w": w})

    assert out.loss.ndim == 0
    assert out.loss.item() == pytest.approx(5.0)

    assert "dummy_int/loss" in out.details
    assert "dummy_int/weighted_loss" in out.details
    assert "dummy_int/mean" in out.details

    assert "dummy_ctx/loss" in out.details
    assert "dummy_ctx/weighted_loss" in out.details
    assert "dummy_ctx/w2" in out.details


def test_composite_scalar_requirement_enforced():
    comp = CompositeObjective(BadNonScalarIntrinsic(), required=True)
    with pytest.raises(ValueError):
        comp(predictions={"p": _rand(3)})


def test_composite_required_flag_requires_at_least_one_required_objective():
    o1 = DummyIntrinsic(required=False)
    o2 = DummyContextual(required=False)

    with pytest.raises(ValueError):
        CompositeObjective(o1, o2, required=True)


def test_composite_passes_empty_dicts_when_none_and_required_objective_raises():
    comp = CompositeObjective(DummyIntrinsic(required=True), required=True)
    with pytest.raises(KeyError):
        comp(predictions=None, targets=None, context=None)


def test_composite_optional_objective_can_skip_when_inputs_missing_and_tensor_exists_somewhere():
    # o1 is required and provides a prediction tensor => composite has a tensor; o2 can skip
    o1 = DummyIntrinsic(required=True, weight=1.0)
    o2 = DummyRelational(required=False, weight=1.0)

    comp = CompositeObjective(o1, o2, required=True)

    p = _rand(4, requires_grad=True)
    out = comp(predictions={"p": p}, targets={})  # o2 missing logits/y, should skip via fallback

    assert out.loss.ndim == 0
    out.loss.backward()
    assert p.grad is not None

def test_base_objective_allows_non_scalar_loss():
    class VectorLossIntrinsic(IntrinsicObjective):
        def __init__(self):
            super().__init__(
                name="vector_loss",
                required=True,
                required_pred_keys=("p",),
                required_context_keys=(),
            )

        def loss(self, predictions, context=None) -> LossOut:
            p = predictions["p"]
            # Return per-sample loss (no reduction)
            return LossOut(loss=p ** 2, details={"per_sample": p.detach()})

    obj = VectorLossIntrinsic()

    x = torch.randn(5, requires_grad=True)
    out = obj(predictions={"p": x})

    # 1) Must be tensor
    assert isinstance(out.loss, torch.Tensor)

    # 2) Must be non-scalar
    assert out.loss.ndim == 1
    assert out.loss.shape == (5,)

    # 3) Backward should work (user may reduce later)
    out.loss.sum().backward()
    assert x.grad is not None

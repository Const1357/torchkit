# tests/test_objectives_impls.py
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from sktorch.modules.nn.objectives.contextual import L2Penalty
from sktorch.modules.nn.objectives.intrinsic import EntropyTerm
from sktorch.modules.nn.objectives.relational import CrossEntropyLoss, MSELoss


# -----------------------
# helpers
# -----------------------

def _randn(*shape: int, requires_grad: bool = False) -> Tensor:
    x = torch.randn(shape, dtype=torch.float32)
    x.requires_grad_(requires_grad)
    return x


# -----------------------
# contextual.py: L2Penalty
# -----------------------

def test_l2penalty_returns_scalar_loss_and_details():
    p1 = _randn(3, 4, requires_grad=True)
    p2 = _randn(5, requires_grad=True)  # bias-like (1D)
    obj = L2Penalty(weight=1.0, include_bias=False)

    out = obj(context={"params": [p1, p2]})

    assert isinstance(out.loss, Tensor)
    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)

    assert out.details["num_tensors"] == 1
    assert out.details["include_bias"] is False
    assert out.details["key"] == "params"

    expected = (p1 * p1).sum()
    assert torch.allclose(out.loss, expected)


def test_l2penalty_include_bias_includes_1d_params():
    p1 = _randn(2, 3, requires_grad=True)
    p2 = _randn(3, requires_grad=True)  # 1D
    obj = L2Penalty(include_bias=True)

    out = obj(context={"params": [p1, p2]})
    expected = (p1 * p1).sum() + (p2 * p2).sum()

    assert out.details["num_tensors"] == 2
    assert torch.allclose(out.loss, expected)


def test_l2penalty_raises_if_no_tensors_found_after_filtering():
    # only 1D tensors, include_bias=False => filtered away => error
    p_bias = _randn(5, requires_grad=True)
    obj = L2Penalty(include_bias=False)

    with pytest.raises(ValueError):
        _ = obj(context={"params": [p_bias]})


# -----------------------
# intrinsic.py: EntropyTerm
# -----------------------

def test_entropy_term_maximize_returns_negative_entropy():
    # valid probabilities (N,C)
    probs = torch.tensor(
        [[0.25, 0.25, 0.25, 0.25],
         [0.70, 0.10, 0.10, 0.10]],
        dtype=torch.float32,
        requires_grad=True,
    )

    obj = EntropyTerm(task="clf", direction="maximize")
    out = obj(predictions={"clf/probs": probs})

    assert out.loss.ndim == 0
    assert torch.isfinite(out.loss)

    # entropy computed in objective (mean over batch)
    p = probs.clamp_min(1e-10)
    entropy = -(p * p.log()).sum(dim=1).mean()
    expected = -entropy  # maximize => minimize -entropy

    assert torch.allclose(out.loss, expected)


def test_entropy_term_minimize_returns_positive_entropy():
    probs = torch.tensor(
        [[0.20, 0.30, 0.50],
         [0.10, 0.10, 0.80]],
        dtype=torch.float32,
        requires_grad=True,
    )

    obj = EntropyTerm(task="clf", direction="minimize")
    out = obj(predictions={"clf/probs": probs})

    p = probs.clamp_min(1e-10)
    entropy = -(p * p.log()).sum(dim=1).mean()
    expected = entropy  # minimize => entropy

    assert torch.allclose(out.loss, expected)


def test_entropy_term_clamps_and_is_finite_when_probs_contain_zeros():
    probs = torch.tensor(
        [[1.0, 0.0, 0.0],
         [0.0, 0.5, 0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )

    obj = EntropyTerm(task="clf", direction="maximize")
    out = obj(predictions={"clf/probs": probs})

    assert torch.isfinite(out.loss)


# -----------------------
# relational.py: CrossEntropyLoss, MSELoss
# -----------------------

def test_cross_entropy_loss_matches_torch_mean():
    torch.manual_seed(0)
    logits = _randn(5, 7, requires_grad=True)
    labels = torch.randint(low=0, high=7, size=(5,), dtype=torch.int64)

    obj = CrossEntropyLoss(task="clf", reduction="mean")
    out = obj(predictions={"clf/logits": logits}, targets={"clf/targets": labels})

    expected = F.cross_entropy(logits, labels, reduction="mean")
    assert out.loss.ndim == 0
    assert torch.allclose(out.loss, expected)
    assert out.details["reduction"] == "mean"


def test_cross_entropy_loss_matches_torch_sum():
    torch.manual_seed(0)
    logits = _randn(4, 3, requires_grad=True)
    labels = torch.randint(low=0, high=3, size=(4,), dtype=torch.int64)

    obj = CrossEntropyLoss(task="clf", reduction="sum")
    out = obj(predictions={"clf/logits": logits}, targets={"clf/targets": labels})

    expected = F.cross_entropy(logits, labels, reduction="sum")
    assert torch.allclose(out.loss, expected)
    assert out.details["reduction"] == "sum"


def test_mse_loss_matches_torch_mean():
    torch.manual_seed(0)
    preds = _randn(6, 2, requires_grad=True)
    tgts = _randn(6, 2)

    obj = MSELoss(task="reg", reduction="mean")
    out = obj(predictions={"reg/pred": preds}, targets={"reg/target": tgts})

    expected = F.mse_loss(preds, tgts, reduction="mean")
    assert out.loss.ndim == 0
    assert torch.allclose(out.loss, expected)
    assert out.details["reduction"] == "mean"


def test_mse_loss_matches_torch_sum():
    torch.manual_seed(0)
    preds = _randn(3, 4, requires_grad=True)
    tgts = _randn(3, 4)

    obj = MSELoss(task="reg", reduction="sum")
    out = obj(predictions={"reg/pred": preds}, targets={"reg/target": tgts})

    expected = F.mse_loss(preds, tgts, reduction="sum")
    assert torch.allclose(out.loss, expected)
    assert out.details["reduction"] == "sum"

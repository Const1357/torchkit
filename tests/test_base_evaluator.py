# tests/test_base_evaluator.py
from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import pytest
import torch
from torch import Tensor

from sktorch.modules.evaluation._base import EvalOut, RelationalEvaluator


class _DummyRelEval(RelationalEvaluator):
    """
    Minimal relational evaluator for testing _BaseEvaluator behavior.

    - Requires predictions["p"] and targets["y"].
    - update() accumulates sum(|p - y|) and count.
    - compute_metrics() returns:
        - "mae": mean absolute error (float)
        - "count": number of seen elements (int)
        - "artifact": a non-scalar tensor (to ensure artifacts are allowed)
    """

    def __init__(self, *, name: str = "dummy", selector: Any = "mae", required: bool = True):
        self._sum_abs: float = 0.0
        self._count: int = 0
        super().__init__(
            name=name,
            selector=selector,
            required=required,
            required_pred_keys=("p",),
            required_target_keys=("y",),
        )

    def reset(self) -> None:
        self._sum_abs = 0.0
        self._count = 0

    def update(
        self,
        predictions: Mapping[str, Tensor | None],
        targets: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> None:
        p = predictions["p"]
        y = targets["y"]
        assert p is not None and y is not None
        d = (p - y).abs().detach().cpu()
        self._sum_abs += float(d.sum().item())
        self._count += int(d.numel())

    def compute_metrics(self) -> Dict[str, Any]:
        if self._count == 0:
            return {"mae": float("nan"), "count": 0, "artifact": torch.zeros(2, 2)}
        return {
            "mae": self._sum_abs / self._count,
            "count": self._count,
            "artifact": torch.zeros(2, 2),  # non-scalar artifact is allowed
        }


def test_base_evaluator_required_missing_keys_raises() -> None:
    ev = _DummyRelEval(required=True)

    with pytest.raises(KeyError, match=r"Missing required"):
        ev(predictions={}, targets={"y": torch.tensor([0.0])})

    with pytest.raises(KeyError, match=r"Missing required"):
        ev(predictions={"p": None}, targets={"y": torch.tensor([0.0])})

    with pytest.raises(KeyError, match=r"Missing required"):
        ev(predictions={"p": torch.tensor([0.0])}, targets={"y": None})


def test_base_evaluator_optional_missing_keys_skips_no_state_change() -> None:
    ev = _DummyRelEval(required=False)
    ev.reset()

    # First, do a valid update to set state.
    _ = ev(predictions={"p": torch.tensor([1.0, 2.0])}, targets={"y": torch.tensor([1.0, 0.0])})
    out1 = ev.compute()
    assert isinstance(out1, EvalOut)
    assert out1.metrics["count"] == 2

    # Now call with missing keys; should skip and NOT change state.
    flat = ev(predictions={"p": torch.tensor([1.0])}, targets={})  # missing y
    assert math.isnan(float(flat["dummy"]))  # selector is NaN on skip
    assert "dummy/missing_target_keys" in flat

    out2 = ev.compute()
    assert out2.metrics["count"] == 2  # unchanged


def test_base_evaluator_selector_single_key() -> None:
    ev = _DummyRelEval(selector="mae", required=True)
    ev.reset()

    flat = ev(predictions={"p": torch.tensor([1.0, 2.0])}, targets={"y": torch.tensor([0.0, 0.0])})
    # mae = (|1| + |2|) / 2 = 1.5
    sel = float(flat["dummy"])
    assert abs(sel - 1.5) < 1e-8
    assert abs(float(flat["dummy/mae"]) - 1.5) < 1e-8
    assert flat["dummy/count"] == 2
    assert isinstance(flat["dummy/artifact"], Tensor)
    assert flat["dummy/artifact"].shape == (2, 2)


def test_base_evaluator_selector_weighted_mixture_renormalizes_and_skips_nan_terms() -> None:
    # mixture uses "mae" and "missing_metric"; missing term should be ignored,
    # weights renormalized => selector == mae
    ev = _DummyRelEval(selector=(("mae", 0.5), ("missing_metric", 0.5)), required=True)
    ev.reset()

    flat = ev(predictions={"p": torch.tensor([2.0])}, targets={"y": torch.tensor([0.0])})
    assert abs(float(flat["dummy"]) - 2.0) < 1e-8

    # mixture with a NaN metric should skip that term too.
    # Force NaN by empty state: compute() after reset without updates.
    ev.reset()
    out = ev.compute()
    assert math.isnan(float(out.selector))


def test_base_evaluator_selector_invalid_specs_rejected() -> None:
    with pytest.raises(ValueError):
        _DummyRelEval(selector="")  # empty key

    with pytest.raises(TypeError):
        _DummyRelEval(selector=123)  # not str / sequence

    with pytest.raises(ValueError):
        _DummyRelEval(selector=())  # empty weighted spec

    with pytest.raises(TypeError):
        _DummyRelEval(selector=(("mae", 1.0, "oops"),))  # wrong tuple length

    with pytest.raises(ValueError):
        _DummyRelEval(selector=(("mae", -1.0),))  # negative weight

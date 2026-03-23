from __future__ import annotations

import pytest
import torch
from torch import Tensor

from torchkit.objectives._base import Objective
from torchkit.objectives.Multitask import MultitaskObjective
from torchkit.objectives.relational import (
    BCELoss,
    CELoss,
    MSELoss,
    DiceLoss,
    SoftDiceLoss,
)


# -------------------------
# Dummy objectives for base tests
# -------------------------

class DummyObjective(Objective):
    def __init__(self, *, is_optional: bool = False):
        super().__init__(
            name="dummy",
            weight=1.0,
            reduction="mean",
            is_optional=is_optional,
        )
        self._required_keys = ("a/b",)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return self._required_keys

    def loss(self, *, inputs: dict[str, Tensor]) -> Tensor:
        x = self.resolve(inputs, "a/b")
        return x.mean()


class NonScalarObjective(Objective):
    def __init__(self):
        super().__init__(name="nonscalar", reduction="mean")
        self._required_keys = ("x",)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return self._required_keys

    def loss(self, *, inputs: dict[str, Tensor]) -> Tensor:
        return self.resolve(inputs, "x")


class BadReturnObjective(Objective):
    def __init__(self):
        super().__init__(name="badreturn", reduction="mean")
        self._required_keys = ("x",)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return self._required_keys

    def loss(self, *, inputs: dict[str, Tensor]):
        return "not a tensor"


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def nested_inputs() -> dict[str, object]:
    return {
        "a": {
            "b": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        },
        "x": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }


@pytest.fixture
def ce_inputs() -> dict[str, object]:
    return {
        "clf": {
            "logits": torch.tensor(
                [
                    [2.0, 0.5, -1.0],
                    [0.1, 1.2, -0.4],
                    [-0.3, 0.7, 1.5],
                    [1.1, -0.2, 0.0],
                ],
                dtype=torch.float32,
            )
        },
        "batch": {
            "y": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        },
    }


@pytest.fixture
def bce_inputs() -> dict[str, object]:
    return {
        "clf": {
            "probabilities": torch.tensor(
                [[0.9], [0.2], [0.8], [0.1]],
                dtype=torch.float32,
            )
        },
        "batch": {
            "target": torch.tensor(
                [[1.0], [0.0], [1.0], [0.0]],
                dtype=torch.float32,
            )
        },
    }


@pytest.fixture
def mse_inputs() -> dict[str, object]:
    return {
        "reg": {
            "predictions": torch.tensor(
                [[1.0], [2.5], [3.0]],
                dtype=torch.float32,
            )
        },
        "batch": {
            "target": torch.tensor(
                [[1.5], [2.0], [2.5]],
                dtype=torch.float32,
            )
        },
    }


@pytest.fixture
def binary_dice_inputs() -> dict[str, object]:
    logits = torch.tensor(
        [
            [[[[2.0, -2.0], [2.0, -2.0]]]],
            [[[[1.0, 1.0], [-1.0, -1.0]]]],
        ],
        dtype=torch.float32,
    )  # (B=2, C=1, D=1, H=2, W=2)

    mask = torch.tensor(
        [
            [[[[1.0, 0.0], [1.0, 0.0]]]],
            [[[[1.0, 1.0], [0.0, 0.0]]]],
        ],
        dtype=torch.float32,
    )

    return {
        "seg": {"logits": logits},
        "batch": {"mask": mask},
    }


@pytest.fixture
def multiclass_dice_inputs() -> dict[str, object]:
    logits = torch.tensor(
        [
            [
                [[[4.0, -2.0], [-2.0, -2.0]]],
                [[[-2.0, 4.0], [-2.0, -2.0]]],
                [[[-2.0, -2.0], [4.0, 4.0]]],
            ]
        ],
        dtype=torch.float32,
    )  # (1,3,1,2,2)

    target = torch.tensor(
        [[[[0, 1], [2, 2]]]],
        dtype=torch.long,
    )  # (1,1,2,2)

    return {
        "seg": {"logits": logits},
        "batch": {"mask": target},
    }


@pytest.fixture
def soft_dice_inputs() -> dict[str, object]:
    logits = torch.tensor(
        [
            [
                [[[4.0, -1.0], [-1.0, -1.0]]],
                [[[-1.0, 4.0], [-1.0, -1.0]]],
                [[[-1.0, -1.0], [4.0, 4.0]]],
            ],
            [
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[0.5, 0.5], [0.5, 0.5]]],
                [[[0.2, 0.2], [0.2, 0.2]]],
            ],
        ],
        dtype=torch.float32,
    )  # (2,3,1,2,2)

    target = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 2.0]]],
            [[[float("nan"), float("nan")], [float("nan"), float("nan")]]],
        ],
        dtype=torch.float32,
    )  # (2,1,2,2)

    return {
        "seg": {"logits": logits},
        "batch": {"target": target},
    }


# -------------------------
# Base Objective tests
# -------------------------

def test_objective_resolve_success(nested_inputs: dict[str, object]):
    x = Objective.resolve(nested_inputs, "a/b")
    assert torch.equal(x, torch.tensor([1.0, 2.0, 3.0]))


def test_objective_resolve_missing_key_raises(nested_inputs: dict[str, object]):
    with pytest.raises(KeyError, match="Key 'missing' not found"):
        Objective.resolve(nested_inputs, "a/missing")


def test_objective_resolve_non_dict_midpath_raises(nested_inputs: dict[str, object]):
    with pytest.raises(TypeError, match="Expected a dict at path a/b"):
        Objective.resolve(nested_inputs, "a/b/c")


def test_objective_resolve_none_raises():
    with pytest.raises(ValueError, match="is None"):
        Objective.resolve({"a": {"b": None}}, "a/b")


def test_objective_resolve_nontensor_raises():
    with pytest.raises(TypeError, match="must be a Tensor"):
        Objective.resolve({"a": {"b": 123}}, "a/b")


def test_objective_forward_runs_loss(nested_inputs: dict[str, object]):
    obj = DummyObjective()
    loss = obj(inputs=nested_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_objective_forward_rejects_none_for_required():
    obj = DummyObjective(is_optional=False)

    with pytest.raises(TypeError, match="must not be None"):
        obj(inputs=None)


def test_objective_forward_optional_none_returns_zero():
    obj = DummyObjective(is_optional=True)

    loss = obj(inputs=None)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert float(loss.item()) == 0.0


def test_objective_forward_missing_required_key_raises():
    obj = DummyObjective(is_optional=False)

    with pytest.raises(KeyError, match="missing required keys"):
        obj(inputs={"a": {}})


def test_objective_forward_optional_missing_key_returns_zero():
    obj = DummyObjective(is_optional=True)

    loss = obj(inputs={"a": {}})

    assert loss.ndim == 0
    assert float(loss.item()) == 0.0


def test_objective_forward_rejects_bad_loss_return_type():
    obj = BadReturnObjective()

    with pytest.raises(TypeError, match="must return a Tensor"):
        obj(inputs={"x": torch.tensor([1.0])})


def test_objective_forward_rejects_nonscalar_loss():
    obj = NonScalarObjective()

    with pytest.raises(ValueError, match="must return a scalar"):
        obj(inputs={"x": torch.tensor([1.0, 2.0])})


# -------------------------
# Relational objectives
# -------------------------

def test_ce_loss_runs(ce_inputs: dict[str, object]):
    obj = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    loss = obj(inputs=ce_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ce_loss_matches_pytorch(ce_inputs: dict[str, object]):
    obj = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    loss = obj(inputs=ce_inputs)
    expected = torch.nn.functional.cross_entropy(
        ce_inputs["clf"]["logits"],
        ce_inputs["batch"]["y"],
        reduction="mean",
    )

    assert torch.allclose(loss, expected)


def test_bce_loss_runs(bce_inputs: dict[str, object]):
    obj = BCELoss(
        input_path="clf/probabilities",
        target_path="batch/target",
        reduction="mean",
    )

    loss = obj(inputs=bce_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_bce_loss_matches_pytorch(bce_inputs: dict[str, object]):
    obj = BCELoss(
        input_path="clf/probabilities",
        target_path="batch/target",
        reduction="mean",
    )

    loss = obj(inputs=bce_inputs)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        bce_inputs["clf"]["probabilities"],
        bce_inputs["batch"]["target"],
        reduction="mean",
    )

    assert torch.allclose(loss, expected)


def test_bce_loss_supports_pos_weight(bce_inputs: dict[str, object]):
    pos_weight = torch.tensor([2.5], dtype=torch.float32)
    obj = BCELoss(
        input_path="clf/probabilities",
        target_path="batch/target",
        pos_weight=pos_weight,
        reduction="mean",
    )

    loss = obj(inputs=bce_inputs)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        bce_inputs["clf"]["probabilities"],
        bce_inputs["batch"]["target"],
        pos_weight=pos_weight,
        reduction="mean",
    )

    assert torch.allclose(loss, expected)


def test_mse_loss_runs(mse_inputs: dict[str, object]):
    obj = MSELoss(
        input_path="reg/predictions",
        target_path="batch/target",
        reduction="mean",
    )

    loss = obj(inputs=mse_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_mse_loss_matches_pytorch(mse_inputs: dict[str, object]):
    obj = MSELoss(
        input_path="reg/predictions",
        target_path="batch/target",
        reduction="mean",
    )

    loss = obj(inputs=mse_inputs)
    expected = torch.nn.functional.mse_loss(
        mse_inputs["reg"]["predictions"],
        mse_inputs["batch"]["target"],
        reduction="mean",
    )

    assert torch.allclose(loss, expected)


def test_relational_required_keys_preserve_user_paths():
    ce = CELoss(input_path="clf/logits", target_path="batch/y")
    bce = BCELoss(input_path="clf/probabilities", target_path="batch/target")
    mse = MSELoss(input_path="reg/predictions", target_path="teacher/value")

    assert ce.required_keys == ("clf/logits", "batch/y")
    assert bce.required_keys == ("clf/probabilities", "batch/target")
    assert mse.required_keys == ("reg/predictions", "teacher/value")


# -------------------------
# MultitaskObjective
# -------------------------

def test_multitask_requires_at_least_one_objective():
    with pytest.raises(ValueError, match="At least one objective must be provided"):
        MultitaskObjective(name="multi")


def test_multitask_rejects_non_objective():
    with pytest.raises(TypeError, match="All objectives must derive from Objective"):
        MultitaskObjective("bad", name="multi")  # type: ignore[arg-type]


def test_multitask_rejects_mismatched_reduction():
    obj1 = MSELoss("reg/predictions", "batch/target", reduction="mean")
    obj2 = MSELoss("reg/predictions", "batch/target", reduction="sum")

    with pytest.raises(ValueError, match="same reduction"):
        MultitaskObjective(obj1, obj2, name="multi")


def test_multitask_sums_weighted_losses(ce_inputs: dict[str, object], mse_inputs: dict[str, object]):
    inputs = {
        "clf": ce_inputs["clf"],
        "reg": mse_inputs["reg"],
        "batch": {
            "y": ce_inputs["batch"]["y"],
            "target": mse_inputs["batch"]["target"],
        },
    }

    ce = CELoss("clf/logits", "batch/y", weight=2.0)
    mse = MSELoss("reg/predictions", "batch/target", weight=0.5)

    multi = MultitaskObjective(ce, mse, name="multi")
    loss = multi(inputs=inputs)

    expected = 2.0 * ce(inputs=inputs) + 0.5 * mse(inputs=inputs)
    assert torch.allclose(loss, expected)

    assert "cross_entropy_loss" in multi.per_objective_loss
    assert "mean_squared_error_loss" in multi.per_objective_loss


def test_multitask_forward_allows_optional_objectives_to_zero_out(mse_inputs: dict[str, object]):
    required = MSELoss("reg/predictions", "batch/target", is_optional=False)
    optional = CELoss("clf/logits", "batch/y", is_optional=True)

    multi = MultitaskObjective(required, optional, name="multi")

    loss = multi(inputs=mse_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


# -------------------------
# DiceLoss
# -------------------------

def test_dice_loss_binary_runs(binary_dice_inputs: dict[str, object]):
    obj = DiceLoss(
        logits_path="seg/logits",
        mask_path="batch/mask",
        reduction="mean",
        is_optional=False,
    )

    loss = obj(inputs=binary_dice_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_dice_loss_multiclass_runs(multiclass_dice_inputs: dict[str, object]):
    obj = DiceLoss(
        logits_path="seg/logits",
        mask_path="batch/mask",
        reduction="mean",
        is_optional=False,
        include_background=True,
    )

    loss = obj(inputs=multiclass_dice_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_dice_loss_optional_all_nan_masks_returns_zero():
    logits = torch.randn(2, 1, 1, 2, 2)
    mask = torch.full((2, 1, 1, 2, 2), float("nan"))

    obj = DiceLoss(
        logits_path="seg/logits",
        mask_path="batch/mask",
        is_optional=True,
    )

    loss = obj(inputs={"seg": {"logits": logits}, "batch": {"mask": mask}})
    assert float(loss.item()) == 0.0


def test_dice_loss_required_all_nan_masks_raises():
    logits = torch.randn(2, 1, 1, 2, 2)
    mask = torch.full((2, 1, 1, 2, 2), float("nan"))

    obj = DiceLoss(
        logits_path="seg/logits",
        mask_path="batch/mask",
        is_optional=False,
    )

    with pytest.raises(ValueError, match="no valid masks"):
        obj(inputs={"seg": {"logits": logits}, "batch": {"mask": mask}})


# -------------------------
# SoftDiceLoss
# -------------------------

def test_soft_dice_loss_runs(soft_dice_inputs: dict[str, object]):
    obj = SoftDiceLoss(
        logits_path="seg/logits",
        target_path="batch/target",
        is_optional=True,
    )

    loss = obj(inputs=soft_dice_inputs)

    assert isinstance(loss, Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert float(loss.item()) >= 0.0


def test_soft_dice_loss_optional_all_nan_returns_zero():
    logits = torch.randn(2, 3, 1, 2, 2)
    target = torch.full((2, 1, 2, 2), float("nan"))

    obj = SoftDiceLoss(
        logits_path="seg/logits",
        target_path="batch/target",
        is_optional=True,
    )

    loss = obj(inputs={"seg": {"logits": logits}, "batch": {"target": target}})
    assert float(loss.item()) == 0.0


def test_soft_dice_loss_required_all_nan_raises():
    logits = torch.randn(2, 3, 1, 2, 2)
    target = torch.full((2, 1, 2, 2), float("nan"))

    obj = SoftDiceLoss(
        logits_path="seg/logits",
        target_path="batch/target",
        is_optional=False,
    )

    with pytest.raises(ValueError, match="no valid masks"):
        obj(inputs={"seg": {"logits": logits}, "batch": {"target": target}})


def test_soft_dice_loss_rejects_out_of_range_targets():
    logits = torch.randn(1, 3, 1, 2, 2)
    target = torch.tensor([[[[0.0, 1.0], [2.0, 5.0]]]], dtype=torch.float32)

    obj = SoftDiceLoss(
        logits_path="seg/logits",
        target_path="batch/target",
        is_optional=False,
    )

    with pytest.raises(ValueError, match="outside \\[0, 2\\]"):
        obj(inputs={"seg": {"logits": logits}, "batch": {"target": target}})

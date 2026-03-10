from __future__ import annotations

import pytest
import torch
from torch import Tensor

from torchkit.models.head_module.classification import (
    ClassifierHeadLinear,
    ClassifierHeadMLP,
)
from torchkit.models.head_module.regression import (
    RegressorHeadLinear,
    RegressorHeadMLP,
)
from torchkit.models.head_module.factory import (
    HeadModuleFactory,
    HeadModuleSpec,
)


@pytest.fixture
def x_2d() -> Tensor:
    return torch.randn(4, 8)


# -------------------------
# Classification heads
# -------------------------

def test_classifier_head_linear_multiclass_logits_only(x_2d: Tensor):
    head = ClassifierHeadLinear(input_dim=8, num_classes=3)

    out = head(x_2d)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (4, 3)


def test_classifier_head_linear_binary_two_logit_logits_only(x_2d: Tensor):
    head = ClassifierHeadLinear(input_dim=8, num_classes=2)

    out = head(x_2d)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (4, 2)


def test_classifier_head_mlp_requires_hidden_layer():
    with pytest.raises(ValueError, match="requires at least one hidden layer"):
        ClassifierHeadMLP(
            input_dim=8,
            hidden_dims=[],
            num_classes=3,
        )


def test_classifier_head_mlp_logits_only(x_2d: Tensor):
    head = ClassifierHeadMLP(
        input_dim=8,
        hidden_dims=[16, 12],
        num_classes=3,
    )

    out = head(x_2d)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (4, 3)


def test_classifier_head_mlp_binary_single_logit_logits_only(x_2d: Tensor):
    head = ClassifierHeadMLP(
        input_dim=8,
        hidden_dims=[16],
        num_classes=1,
    )

    out = head(x_2d)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (4, 1)


def test_classifier_head_mlp_multiclass_logits_only(x_2d: Tensor):
    head = ClassifierHeadMLP(
        input_dim=8,
        hidden_dims=[16],
        num_classes=4,
    )

    out = head(x_2d)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"logits"}
    assert out["logits"].shape == (4, 4)


# -------------------------
# Regression heads
# -------------------------

def test_regressor_head_linear_default_target_count(x_2d: Tensor):
    head = RegressorHeadLinear(input_dim=8)

    out = head(x_2d)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"predictions"}
    assert out["predictions"].shape == (4, 1)


def test_regressor_head_linear_multiple_targets(x_2d: Tensor):
    head = RegressorHeadLinear(input_dim=8, n_targets=3)

    out = head(x_2d)

    assert set(out.keys()) == {"predictions"}
    assert out["predictions"].shape == (4, 3)


def test_regressor_head_linear_rejects_nonpositive_targets():
    with pytest.raises(ValueError, match="must be positive"):
        RegressorHeadLinear(input_dim=8, n_targets=0)


def test_regressor_head_mlp_requires_hidden_layer():
    with pytest.raises(ValueError, match="requires at least one hidden layer"):
        RegressorHeadMLP(
            input_dim=8,
            hidden_dims=[],
            n_targets=1,
        )


def test_regressor_head_mlp_rejects_nonpositive_targets():
    with pytest.raises(ValueError, match="must be positive"):
        RegressorHeadMLP(
            input_dim=8,
            hidden_dims=[16],
            n_targets=0,
        )


def test_regressor_head_mlp_rejects_negative_dropout():
    with pytest.raises(ValueError, match="dropout"):
        RegressorHeadMLP(
            input_dim=8,
            hidden_dims=[16],
            n_targets=1,
            dropout=-0.1,
        )


def test_regressor_head_mlp_forward(x_2d: Tensor):
    head = RegressorHeadMLP(
        input_dim=8,
        hidden_dims=[16, 12],
        n_targets=2,
    )

    out = head(x_2d)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"predictions"}
    assert out["predictions"].shape == (4, 2)


# -------------------------
# Factory
# -------------------------

def test_head_module_factory_builds_classifier():
    spec = HeadModuleSpec(
        cls=ClassifierHeadLinear,
        kwargs={"input_dim": 8, "num_classes": 3},
    )

    module = HeadModuleFactory.build(spec)

    assert isinstance(module, ClassifierHeadLinear)


def test_head_module_factory_builds_regressor():
    spec = HeadModuleSpec(
        cls=RegressorHeadLinear,
        kwargs={"input_dim": 8, "n_targets": 2},
    )

    module = HeadModuleFactory.build(spec)

    assert isinstance(module, RegressorHeadLinear)


def test_head_module_factory_rejects_missing_cls():
    spec = HeadModuleSpec(cls=None)

    with pytest.raises(ValueError, match="must be specified"):
        HeadModuleFactory.build(spec)


def test_head_module_factory_rejects_non_module_cls():
    class NotAModule:
        pass

    spec = HeadModuleSpec(cls=NotAModule)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a subclass of torch.nn.Module"):
        HeadModuleFactory.build(spec)


def test_head_module_factory_rejects_both_state_dict_and_path(tmp_path):
    spec = HeadModuleSpec(
        cls=ClassifierHeadLinear,
        kwargs={"input_dim": 8, "num_classes": 3},
    )
    path = tmp_path / "dummy.pt"
    torch.save({}, path)

    with pytest.raises(ValueError, match="Only one of state_dict_path or state_dict"):
        HeadModuleFactory.build(
            spec,
            state_dict_path=str(path),
            state_dict={},
        )


def test_head_module_factory_can_load_state_dict():
    original = ClassifierHeadLinear(input_dim=8, num_classes=3)
    state_dict = original.state_dict()

    spec = HeadModuleSpec(
        cls=ClassifierHeadLinear,
        kwargs={"input_dim": 8, "num_classes": 3},
    )

    loaded = HeadModuleFactory.build(spec, state_dict=state_dict)

    assert isinstance(loaded, ClassifierHeadLinear)
    for k in state_dict:
        assert torch.allclose(state_dict[k], loaded.state_dict()[k])
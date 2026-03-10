from __future__ import annotations

import pytest
import torch
from torch import nn, Tensor

from torchkit.models.backbone._backbone import Backbone
from torchkit.models.backbone.MLP_backbone import MLPBackbone
from torchkit.models.backbone.factory import BackboneFactory, BackboneSpec


# -------------------------
# Dummy backbone for tests
# -------------------------

class DummyBackbone(Backbone):

    def __init__(self):
        super().__init__(supported_features=["a", "b"])

    def _forward_impl(self, input, *, requested_features=None, **kwargs):
        out = {}
        if "a" in requested_features:
            out["a"] = input["x"] + 1
        if "b" in requested_features:
            out["b"] = input["x"] + 2
        return out


class BadReturnTypeBackbone(Backbone):

    def __init__(self):
        super().__init__(supported_features=["a"])

    def _forward_impl(self, input, *, requested_features=None, **kwargs):
        return "not a dict"


class BadTensorBackbone(Backbone):

    def __init__(self):
        super().__init__(supported_features=["a"])

    def _forward_impl(self, input, *, requested_features=None, **kwargs):
        return {"a": "not a tensor"}


class UnknownKeyBackbone(Backbone):

    def __init__(self):
        super().__init__(supported_features=["a"])

    def _forward_impl(self, input, *, requested_features=None, **kwargs):
        return {"b": torch.zeros(1)}


class MissingRequestedBackbone(Backbone):

    def __init__(self):
        super().__init__(supported_features=["a", "b"])

    def _forward_impl(self, input, *, requested_features=None, **kwargs):
        return {"a": torch.zeros(1)}


# -------------------------
# Fixtures
# -------------------------

@pytest.fixture
def input_dict():
    return {"x": torch.randn(4, 8)}


# -------------------------
# Base backbone behavior
# -------------------------

def test_backbone_available_features():
    backbone = DummyBackbone()
    assert set(backbone.available_features) == {"a", "b"}


def test_backbone_requested_features_subset(input_dict):
    backbone = DummyBackbone()

    out = backbone(input_dict, requested_features=["a"])

    assert set(out.keys()) == {"a"}


def test_backbone_requested_features_validation_type(input_dict):
    backbone = DummyBackbone()

    with pytest.raises(TypeError):
        backbone(input_dict, requested_features=123)


def test_backbone_requested_features_unknown_key(input_dict):
    backbone = DummyBackbone()

    with pytest.raises(KeyError):
        backbone(input_dict, requested_features=["c"])


def test_backbone_forward_returns_all_requested(input_dict):
    backbone = DummyBackbone()

    out = backbone(input_dict, requested_features=["a", "b"])

    assert set(out.keys()) == {"a", "b"}


# -------------------------
# Output validation
# -------------------------

def test_backbone_rejects_non_dict_output(input_dict):
    backbone = BadReturnTypeBackbone()

    with pytest.raises(TypeError):
        backbone(input_dict)


def test_backbone_rejects_non_tensor_outputs(input_dict):
    backbone = BadTensorBackbone()

    with pytest.raises(TypeError):
        backbone(input_dict)


def test_backbone_rejects_unknown_feature_keys(input_dict):
    backbone = UnknownKeyBackbone()

    with pytest.raises(KeyError):
        backbone(input_dict)


def test_backbone_requires_requested_features_returned(input_dict):
    backbone = MissingRequestedBackbone()

    with pytest.raises(KeyError):
        backbone(input_dict, requested_features=["a", "b"])


# -------------------------
# Freeze / unfreeze
# -------------------------

def test_backbone_freeze_unfreeze():
    backbone = MLPBackbone(
        input_dim=8,
        hidden_dims=[16],
        output_dim=4,
    )

    backbone.freeze()

    for p in backbone.parameters():
        assert p.requires_grad is False

    backbone.unfreeze()

    for p in backbone.parameters():
        assert p.requires_grad is True


# -------------------------
# MLP backbone behavior
# -------------------------

def test_mlp_backbone_forward_shape():
    backbone = MLPBackbone(
        input_dim=8,
        hidden_dims=[16, 16],
        output_dim=5,
    )

    x = torch.randn(3, 8)

    out = backbone({"x": x})

    assert "features" in out
    assert out["features"].shape == (3, 5)


def test_mlp_backbone_requested_features():
    backbone = MLPBackbone(
        input_dim=8,
        hidden_dims=[16],
        output_dim=4,
    )

    x = torch.randn(2, 8)

    out = backbone({"x": x}, requested_features=["features"])

    assert list(out.keys()) == ["features"]


# -------------------------
# Factory
# -------------------------

def test_backbone_factory_build():
    spec = BackboneSpec(
        cls=MLPBackbone,
        kwargs=dict(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
        ),
    )

    backbone = BackboneFactory.build(spec)

    assert isinstance(backbone, MLPBackbone)


def test_backbone_factory_rejects_missing_cls():
    spec = BackboneSpec(cls=None)

    with pytest.raises(ValueError):
        BackboneFactory.build(spec)


def test_backbone_factory_rejects_non_backbone_cls():
    class NotBackbone(nn.Module):
        pass

    spec = BackboneSpec(cls=NotBackbone)

    with pytest.raises(TypeError):
        BackboneFactory.build(spec)


def test_backbone_factory_state_dict_loading():
    backbone = MLPBackbone(
        input_dim=8,
        hidden_dims=[16],
        output_dim=4,
    )

    state_dict = backbone.state_dict()

    spec = BackboneSpec(
        cls=MLPBackbone,
        kwargs=dict(
            input_dim=8,
            hidden_dims=[16],
            output_dim=4,
        ),
    )

    loaded = BackboneFactory.build(spec, state_dict=state_dict)

    for k in state_dict:
        assert torch.allclose(state_dict[k], loaded.state_dict()[k])

def test_backbone_default_returns_all_supported_features(input_dict):
    backbone = DummyBackbone()

    out = backbone(input_dict)

    assert set(out.keys()) == {"a", "b"}


@pytest.mark.parametrize(
    "requested_features",
    [
        ["a"],
        ["b"],
        ["a", "b"],
        ("a", "b"),
        {"a", "b"},
    ],
)
def test_backbone_returns_exact_requested_feature_subset(input_dict, requested_features):
    backbone = DummyBackbone()

    out = backbone(input_dict, requested_features=requested_features)

    assert set(out.keys()) == set(requested_features)


def test_backbone_requested_feature_order_does_not_matter(input_dict):
    backbone = DummyBackbone()

    out1 = backbone(input_dict, requested_features=["a", "b"])
    out2 = backbone(input_dict, requested_features=["b", "a"])

    assert set(out1.keys()) == {"a", "b"}
    assert set(out2.keys()) == {"a", "b"}
    assert torch.equal(out1["a"], out2["a"])
    assert torch.equal(out1["b"], out2["b"])


def test_backbone_empty_requested_features_returns_empty_dict(input_dict):
    backbone = DummyBackbone()

    out = backbone(input_dict, requested_features=[])

    assert out == {}


class IgnoresRequestedFeaturesBackbone(Backbone):
    def __init__(self):
        super().__init__(supported_features=["a", "b"])

    def _forward_impl(self, input, *, requested_features=None, **kwargs):
        # Bad implementation: always returns all features
        return {
            "a": input["x"] + 1,
            "b": input["x"] + 2,
        }


def test_backbone_rejects_unrequested_features_under_strict_contract(input_dict):
    backbone = IgnoresRequestedFeaturesBackbone()

    with pytest.raises(KeyError, match="returned unrequested features"):
        backbone(input_dict, requested_features=["a"])
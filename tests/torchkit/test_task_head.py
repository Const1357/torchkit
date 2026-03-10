from __future__ import annotations

import pytest
import torch
from torch import nn, Tensor

from torchkit.models.head._task_head import TaskHead
from torchkit.models.head.factory import TaskHeadFactory, TaskHeadSpec

from torchkit.models.fuse._fuse_module import FuseModule
from torchkit.models.fuse.factory import FuseModuleSpec

from torchkit.models.adapters._feature_adapter import FeatureAdapter
from torchkit.models.adapters.factory import FeatureAdapterSpec

from torchkit.models.head_module.factory import HeadModuleSpec


class DummyFuse(FuseModule):
    def forward(self, features: dict[str, Tensor], **kwargs) -> Tensor:
        return torch.cat([features[k] for k in sorted(features.keys())], dim=1)


class DummyFuseWithPayload(FuseModule):
    def __init__(self):
        super().__init__()
        self.last_payload = None

    def forward(self, features: dict[str, Tensor], *, payload=None, **kwargs) -> Tensor:
        self.last_payload = payload
        return torch.cat([features[k] for k in sorted(features.keys())], dim=1)


class DummyBadFuse(FuseModule):
    def forward(self, features: dict[str, Tensor], **kwargs):
        return {"not": "a tensor"}


class DummyAdapter(FeatureAdapter):
    def forward(self, features: Tensor, **kwargs) -> Tensor:
        return features + 1.0


class DummyAdapterWithPayload(FeatureAdapter):
    def __init__(self):
        super().__init__()
        self.last_payload = None

    def forward(self, features: Tensor, *, payload=None, **kwargs) -> Tensor:
        self.last_payload = payload
        return features + 1.0


class DummyHeadModule(nn.Module):
    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        return {"logits": x}


class DummyHeadModuleWithPayload(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_payload = None

    def forward(self, x: Tensor, *, payload=None, **kwargs) -> dict[str, Tensor]:
        self.last_payload = payload
        return {"logits": x}


class DummyNoneHeadModule(nn.Module):
    def forward(self, x: Tensor, **kwargs):
        return None


class StatefulAdapter(FeatureAdapter):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, features: Tensor, **kwargs) -> Tensor:
        return features * self.scale


@pytest.fixture
def features() -> dict[str, Tensor]:
    return {
        "f1": torch.randn(4, 3),
        "f2": torch.randn(4, 5),
        "unused": torch.randn(4, 7),
    }


@pytest.fixture
def payload() -> dict[str, Tensor]:
    return {
        "x": torch.randn(4, 10),
        "tabular": torch.randn(4, 2),
    }


def test_task_head_requires_required_features():
    with pytest.raises(ValueError, match="must specify required_features"):
        TaskHead(required_features=None)


def test_task_head_rejects_bad_required_features_type():
    with pytest.raises(TypeError, match="must be a str or one of"):
        TaskHead(required_features=123)


def test_task_head_rejects_empty_required_features():
    with pytest.raises(ValueError, match="must be non-empty"):
        TaskHead(required_features=[])


def test_task_head_rejects_nonstring_required_features():
    with pytest.raises(TypeError, match="must contain only str"):
        TaskHead(required_features=["f1", 123])


def test_task_head_requires_fuse_module_for_multiple_features():
    with pytest.raises(ValueError, match="must be provided .* more than one feature"):
        TaskHead(
            required_features={"f1", "f2"},
            fuse_module=None,
            feature_adapter=DummyAdapter(),
            head_module=DummyHeadModule(),
        )


def test_task_head_ignores_fuse_module_for_single_string_feature():
    fuse = DummyFuse()

    with pytest.warns(UserWarning, match="Ignoring `fuse_module`"):
        head = TaskHead(
            required_features="f1",
            fuse_module=fuse,
            feature_adapter=DummyAdapter(),
            head_module=DummyHeadModule(),
        )

    assert head.fuse_module is None


def test_task_head_defaults_to_identity_modules():
    with pytest.warns(UserWarning):
        head = TaskHead(required_features="f1")

    assert isinstance(head.feature_adapter, nn.Identity)
    assert isinstance(head.head_module, nn.Identity)


def test_task_head_is_active_by_default():
    head = TaskHead(
        required_features="f1",
        feature_adapter=DummyAdapter(),
        head_module=DummyHeadModule(),
    )
    assert head.is_active is True


def test_task_head_enable_disable():
    head = TaskHead(
        required_features="f1",
        feature_adapter=DummyAdapter(),
        head_module=DummyHeadModule(),
        active=False,
    )

    assert head.is_active is False
    assert head.enable() is head
    assert head.is_active is True
    assert head.disable() is head
    assert head.is_active is False


def test_task_head_inactive_returns_none(features: dict[str, Tensor]):
    head = TaskHead(
        required_features="f1",
        feature_adapter=DummyAdapter(),
        head_module=DummyHeadModule(),
        active=False,
    )

    out = head(features)
    assert out is None


def test_task_head_missing_required_features_raises(features: dict[str, Tensor]):
    head = TaskHead(
        required_features={"f1", "missing"},
        fuse_module=DummyFuse(),
        feature_adapter=DummyAdapter(),
        head_module=DummyHeadModule(),
    )

    with pytest.raises(KeyError, match="missing required backbone features"):
        head(features)


def test_task_head_single_feature_no_fuse(features: dict[str, Tensor]):
    head = TaskHead(
        required_features="f1",
        feature_adapter=DummyAdapter(),
        head_module=DummyHeadModule(),
    )

    out = head(features)

    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape == features["f1"].shape
    assert torch.allclose(out["logits"], features["f1"] + 1.0)


def test_task_head_multiple_features_with_fuse(features: dict[str, Tensor]):
    head = TaskHead(
        required_features={"f1", "f2"},
        fuse_module=DummyFuse(),
        feature_adapter=DummyAdapter(),
        head_module=DummyHeadModule(),
    )

    out = head(features)

    assert isinstance(out, dict)
    assert "logits" in out
    assert out["logits"].shape == (4, 8)


def test_task_head_uses_only_required_features(features: dict[str, Tensor]):
    head = TaskHead(
        required_features={"f1", "f2"},
        fuse_module=DummyFuse(),
        feature_adapter=DummyAdapter(),
        head_module=DummyHeadModule(),
    )

    out = head(features)

    expected = torch.cat([features["f1"], features["f2"]], dim=1) + 1.0
    assert torch.allclose(out["logits"], expected)


def test_task_head_rejects_nontensor_output_from_fuse(features: dict[str, Tensor]):
    head = TaskHead(
        required_features={"f1", "f2"},
        fuse_module=DummyBadFuse(),
        feature_adapter=DummyAdapter(),
        head_module=DummyHeadModule(),
    )

    with pytest.raises(TypeError, match="After fuse_module, expected a Tensor"):
        head(features)


def test_task_head_payload_forwarded_to_fuse_only_when_supported(
    features: dict[str, Tensor],
    payload: dict[str, Tensor],
):
    fuse = DummyFuseWithPayload()
    head = TaskHead(
        required_features={"f1", "f2"},
        fuse_module=fuse,
        feature_adapter=DummyAdapter(),
        head_module=DummyHeadModule(),
    )

    _ = head(features, payload=payload)
    assert fuse.last_payload is payload


def test_task_head_payload_forwarded_to_feature_adapter_when_supported(
    features: dict[str, Tensor],
    payload: dict[str, Tensor],
):
    adapter = DummyAdapterWithPayload()
    head = TaskHead(
        required_features="f1",
        feature_adapter=adapter,
        head_module=DummyHeadModule(),
    )

    _ = head(features, payload=payload)
    assert adapter.last_payload is payload


def test_task_head_payload_forwarded_to_head_module_when_supported(
    features: dict[str, Tensor],
    payload: dict[str, Tensor],
):
    head_module = DummyHeadModuleWithPayload()
    head = TaskHead(
        required_features="f1",
        feature_adapter=DummyAdapter(),
        head_module=head_module,
    )

    _ = head(features, payload=payload)
    assert head_module.last_payload is payload


def test_task_head_none_output_from_head_module_raises(features: dict[str, Tensor]):
    head = TaskHead(
        required_features="f1",
        feature_adapter=DummyAdapter(),
        head_module=DummyNoneHeadModule(),
    )

    with pytest.raises(RuntimeError, match="produced None output"):
        head(features)


def test_task_head_freeze_unfreeze():
    head = TaskHead(
        required_features="f1",
        feature_adapter=StatefulAdapter(scale=2.0),
        head_module=nn.Linear(3, 2),
    )

    head.freeze()
    for p in head.parameters():
        assert p.requires_grad is False

    head.unfreeze()
    for p in head.parameters():
        assert p.requires_grad is True


def test_task_head_factory_builds_task_head():
    spec = TaskHeadSpec(
        required_features="f1",
        fuse_module=None,
        feature_adapter=FeatureAdapterSpec(cls=StatefulAdapter, kwargs={"scale": 2.0}),
        head_module=HeadModuleSpec(cls=nn.Linear, kwargs={"in_features": 3, "out_features": 2}),
        active=True,
    )

    head = TaskHeadFactory.build(spec)

    assert isinstance(head, TaskHead)
    assert head.is_active is True
    assert head.required_features == {"f1"}
    assert isinstance(head.feature_adapter, StatefulAdapter)
    assert isinstance(head.head_module, nn.Linear)


def test_task_head_factory_rejects_both_whole_state_dict_and_path():
    spec = TaskHeadSpec(
        required_features="f1",
        feature_adapter=FeatureAdapterSpec(cls=StatefulAdapter, kwargs={"scale": 2.0}),
        head_module=HeadModuleSpec(cls=nn.Linear, kwargs={"in_features": 3, "out_features": 2}),
    )

    with pytest.raises(ValueError, match="Only one of state_dict_path or state_dict"):
        TaskHeadFactory.build(
            spec,
            state_dict={},
            state_dict_path="dummy.pt",
        )


def test_task_head_factory_rejects_mixing_whole_and_nested_loading():
    spec = TaskHeadSpec(
        required_features="f1",
        feature_adapter=FeatureAdapterSpec(cls=StatefulAdapter, kwargs={"scale": 2.0}),
        head_module=HeadModuleSpec(cls=nn.Linear, kwargs={"in_features": 3, "out_features": 2}),
    )

    with pytest.raises(ValueError, match="cannot be mixed with nested component state loading"):
        TaskHeadFactory.build(
            spec,
            state_dict={},
            feature_adapter_state_dict={},
        )


def test_task_head_factory_can_load_nested_feature_adapter_state_dict():
    adapter = StatefulAdapter(scale=3.5)
    adapter_sd = adapter.state_dict()

    spec = TaskHeadSpec(
        required_features="f1",
        feature_adapter=FeatureAdapterSpec(cls=StatefulAdapter, kwargs={"scale": 1.0}),
        head_module=HeadModuleSpec(cls=nn.Linear, kwargs={"in_features": 3, "out_features": 2}),
    )

    head = TaskHeadFactory.build(
        spec,
        feature_adapter_state_dict=adapter_sd,
    )

    assert isinstance(head.feature_adapter, StatefulAdapter)
    assert torch.allclose(head.feature_adapter.scale, torch.tensor(3.5))


def test_task_head_factory_can_load_whole_task_head_state_dict():
    original = TaskHead(
        required_features={"f1", "f2"},
        fuse_module=DummyFuse(),
        feature_adapter=StatefulAdapter(scale=2.0),
        head_module=nn.Linear(8, 4),
        active=True,
    )
    state_dict = original.state_dict()

    spec = TaskHeadSpec(
        required_features={"f1", "f2"},
        fuse_module=FuseModuleSpec(cls=DummyFuse, kwargs={}),
        feature_adapter=FeatureAdapterSpec(cls=StatefulAdapter, kwargs={"scale": 1.0}),
        head_module=HeadModuleSpec(cls=nn.Linear, kwargs={"in_features": 8, "out_features": 4}),
        active=True,
    )

    loaded = TaskHeadFactory.build(spec, state_dict=state_dict)

    assert isinstance(loaded, TaskHead)
    for k, v in state_dict.items():
        assert torch.allclose(v, loaded.state_dict()[k])
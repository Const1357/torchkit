from __future__ import annotations

import pytest
import torch
from torch import Tensor

from torchkit.models.fuse._fuse_module import (
    FuseModule,
    ConcatFuseModule,
    SumFuseModule,
    TabularConcatFuseModule,
)
from torchkit.models.fuse.factory import FuseModuleFactory, FuseModuleSpec


class DummyFuseModule(FuseModule):
    def forward(self, features: dict[str, Tensor], **kwargs):
        vals = list(features.values())
        return vals[0]


class StatefulFuseModule(FuseModule):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, features: dict[str, Tensor], **kwargs):
        vals = list(features.values())
        return vals[0] * self.scale


@pytest.fixture
def features_2d() -> dict[str, Tensor]:
    return {
        "a": torch.randn(4, 3),
        "b": torch.randn(4, 5),
    }


@pytest.fixture
def features_same_shape_2d() -> dict[str, Tensor]:
    return {
        "a": torch.randn(4, 3),
        "b": torch.randn(4, 3),
    }


@pytest.fixture
def features_4d() -> dict[str, Tensor]:
    return {
        "a": torch.randn(2, 3, 8, 8),
        "b": torch.randn(2, 5, 8, 8),
    }


@pytest.fixture
def features_same_shape_4d() -> dict[str, Tensor]:
    return {
        "a": torch.randn(2, 3, 8, 8),
        "b": torch.randn(2, 3, 8, 8),
    }


def test_dummy_fuse_module_returns_tensor(features_2d: dict[str, Tensor]):
    module = DummyFuseModule()

    out = module(features_2d)

    assert isinstance(out, torch.Tensor)
    assert out.shape == features_2d["a"].shape


def test_concat_fuse_module_2d(features_2d: dict[str, Tensor]):
    module = ConcatFuseModule(dim=1)

    out = module(features_2d)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 8)


def test_concat_fuse_module_4d(features_4d: dict[str, Tensor]):
    module = ConcatFuseModule(dim=1)

    out = module(features_4d)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8, 8, 8)


def test_concat_fuse_module_rejects_empty_dict():
    module = ConcatFuseModule()

    with pytest.raises(ValueError, match="expects a non-empty dict"):
        module({})


def test_concat_fuse_module_rejects_non_dict():
    module = ConcatFuseModule()

    with pytest.raises(ValueError, match="expects a non-empty dict"):
        module([torch.randn(2, 3)])  # type: ignore[arg-type]


def test_sum_fuse_module_2d(features_same_shape_2d: dict[str, Tensor]):
    module = SumFuseModule(stack_dim=0)

    out = module(features_same_shape_2d)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 3)

    expected = features_same_shape_2d["a"] + features_same_shape_2d["b"]
    assert torch.allclose(out, expected)


def test_sum_fuse_module_4d(features_same_shape_4d: dict[str, Tensor]):
    module = SumFuseModule(stack_dim=0)

    out = module(features_same_shape_4d)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 3, 8, 8)

    expected = features_same_shape_4d["a"] + features_same_shape_4d["b"]
    assert torch.allclose(out, expected)


def test_sum_fuse_module_rejects_empty_dict():
    module = SumFuseModule()

    with pytest.raises(ValueError, match="expects a non-empty dict"):
        module({})


def test_sum_fuse_module_rejects_non_dict():
    module = SumFuseModule()

    with pytest.raises(ValueError, match="expects a non-empty dict"):
        module([torch.randn(2, 3)])  # type: ignore[arg-type]


def test_tabular_concat_fuse_module_2d_single_feature():
    module = TabularConcatFuseModule(tabular_key="tabular", dim=1)

    features = {"a": torch.randn(4, 3)}
    payload = {"tabular": torch.randn(4, 2)}

    out = module(features, payload=payload)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 5)


def test_tabular_concat_fuse_module_2d_multiple_features(features_2d: dict[str, Tensor]):
    module = TabularConcatFuseModule(tabular_key="tabular", dim=1)
    payload = {"tabular": torch.randn(4, 2)}

    out = module(features_2d, payload=payload)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 10)  # 3 + 5 + 2


def test_tabular_concat_fuse_module_4d_broadcasts_tabular(features_4d: dict[str, Tensor]):
    module = TabularConcatFuseModule(tabular_key="tabular", dim=1)
    payload = {"tabular": torch.randn(2, 4)}

    out = module(features_4d, payload=payload)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 12, 8, 8)  # (3 + 5) + 4


def test_tabular_concat_fuse_module_requires_payload(features_2d: dict[str, Tensor]):
    module = TabularConcatFuseModule(tabular_key="tabular")

    with pytest.raises(KeyError, match="payload.*must contain key"):
        module(features_2d)


def test_tabular_concat_fuse_module_requires_payload_key(features_2d: dict[str, Tensor]):
    module = TabularConcatFuseModule(tabular_key="tabular")

    with pytest.raises(KeyError, match="payload.*must contain key"):
        module(features_2d, payload={"other": torch.randn(4, 2)})


def test_tabular_concat_fuse_module_requires_tensor_payload(features_2d: dict[str, Tensor]):
    module = TabularConcatFuseModule(tabular_key="tabular")

    with pytest.raises(TypeError, match="must be a Tensor"):
        module(features_2d, payload={"tabular": [1, 2, 3]})


def test_tabular_concat_fuse_module_rejects_ndim_mismatch(features_2d: dict[str, Tensor]):
    module = TabularConcatFuseModule(tabular_key="tabular")

    # 2D features, but tabular is 4D -> no broadcasting rule here, should fail
    bad_payload = {"tabular": torch.randn(4, 2, 1, 1)}

    with pytest.raises(ValueError, match="ndim must match"):
        module(features_2d, payload=bad_payload)


def test_tabular_concat_fuse_module_rejects_batch_mismatch(features_2d: dict[str, Tensor]):
    module = TabularConcatFuseModule(tabular_key="tabular")

    bad_payload = {"tabular": torch.randn(5, 2)}

    with pytest.raises(ValueError, match="Batch size mismatch"):
        module(features_2d, payload=bad_payload)


def test_tabular_concat_fuse_module_rejects_empty_dict():
    module = TabularConcatFuseModule(tabular_key="tabular")

    with pytest.raises(ValueError, match="expects a non-empty dict"):
        module({}, payload={"tabular": torch.randn(4, 2)})


def test_fuse_module_factory_builds_concat_module():
    spec = FuseModuleSpec(
        cls=ConcatFuseModule,
        kwargs={"dim": 1},
    )

    module = FuseModuleFactory.build(spec)

    assert isinstance(module, ConcatFuseModule)
    assert module.dim == 1


def test_fuse_module_factory_rejects_missing_cls():
    spec = FuseModuleSpec(cls=None)

    with pytest.raises(ValueError, match="must be specified"):
        FuseModuleFactory.build(spec)


def test_fuse_module_factory_rejects_non_fuse_cls():
    class NotAFuseModule:
        pass

    spec = FuseModuleSpec(cls=NotAFuseModule)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a subclass of FuseModule"):
        FuseModuleFactory.build(spec)


def test_fuse_module_factory_rejects_both_state_dict_and_path(tmp_path):
    spec = FuseModuleSpec(cls=ConcatFuseModule, kwargs={"dim": 1})
    path = tmp_path / "dummy.pt"
    torch.save({}, path)

    with pytest.raises(ValueError, match="Only one of state_dict_path or state_dict"):
        FuseModuleFactory.build(
            spec,
            state_dict_path=str(path),
            state_dict={},
        )


def test_fuse_module_factory_can_load_state_dict():
    original = StatefulFuseModule(scale=2.5)
    state_dict = original.state_dict()

    spec = FuseModuleSpec(cls=StatefulFuseModule, kwargs={"scale": 1.0})
    loaded = FuseModuleFactory.build(spec, state_dict=state_dict)

    assert isinstance(loaded, StatefulFuseModule)
    assert torch.allclose(loaded.scale, torch.tensor(2.5))
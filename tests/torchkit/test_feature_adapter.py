from __future__ import annotations

import pytest
import torch
from torch import Tensor

from torchkit.models.adapters._feature_adapter import (
    FeatureAdapter,
    IdentityAdapter,
    FlattenAdapter,
    GAPAdapter,
    GMPAdapter,
    GAPGMPConcatAdapter,
    StatsPoolAdapter,
    GeMAdapter,
    LogSumExpPoolAdapter,
    AttnPoolAdapter,
    SPPAdapter,
    Conv1x1GAPAdapter,
    TokenAdapter,
)
from torchkit.models.adapters.factory import FeatureAdapterFactory, FeatureAdapterSpec


class DummyFeatureAdapter(FeatureAdapter):
    def forward(self, features: Tensor, **kwargs) -> Tensor:
        return features + 1.0


class StatefulFeatureAdapter(FeatureAdapter):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, features: Tensor, **kwargs) -> Tensor:
        return features * self.scale


@pytest.fixture
def x_2d() -> Tensor:
    return torch.randn(4, 6)


@pytest.fixture
def x_3d() -> Tensor:
    return torch.randn(4, 6, 10)


@pytest.fixture
def x_4d() -> Tensor:
    return torch.randn(4, 6, 8, 8)


@pytest.fixture
def x_5d() -> Tensor:
    return torch.randn(2, 6, 5, 6, 7)


def test_dummy_feature_adapter_returns_tensor(x_2d: Tensor):
    adapter = DummyFeatureAdapter()

    out = adapter(x_2d)

    assert isinstance(out, Tensor)
    assert out.shape == x_2d.shape
    assert torch.allclose(out, x_2d + 1.0)


def test_identity_adapter(x_4d: Tensor):
    adapter = IdentityAdapter()

    out = adapter(x_4d)

    assert out.shape == x_4d.shape
    assert torch.equal(out, x_4d)


def test_flatten_adapter(x_4d: Tensor):
    adapter = FlattenAdapter()

    out = adapter(x_4d)

    assert out.shape == (x_4d.shape[0], x_4d[0].numel())


def test_gap_adapter_3d(x_3d: Tensor):
    adapter = GAPAdapter()

    out = adapter(x_3d)

    assert out.shape == (4, 6)
    assert torch.allclose(out, x_3d.mean(dim=(2,)))


def test_gap_adapter_4d_keepdim(x_4d: Tensor):
    adapter = GAPAdapter(keepdim=True)

    out = adapter(x_4d)

    assert out.shape == (4, 6, 1, 1)


def test_gap_adapter_rejects_ndim_lt_3(x_2d: Tensor):
    adapter = GAPAdapter()

    with pytest.raises(ValueError, match="ndim>=3"):
        adapter(x_2d)


def test_gmp_adapter_4d(x_4d: Tensor):
    adapter = GMPAdapter()

    out = adapter(x_4d)

    assert out.shape == (4, 6)
    assert torch.allclose(out, x_4d.amax(dim=(2, 3)))


def test_gapgmp_concat_adapter(x_4d: Tensor):
    adapter = GAPGMPConcatAdapter()

    out = adapter(x_4d)

    assert out.shape == (4, 12)


def test_stats_pool_adapter(x_4d: Tensor):
    adapter = StatsPoolAdapter()

    out = adapter(x_4d)

    assert out.shape == (4, 12)
    assert torch.isfinite(out).all()


def test_gem_adapter_fixed_p(x_4d: Tensor):
    adapter = GeMAdapter(p=3.0, learnable_p=False)

    out = adapter(x_4d.clamp_min(0.1))

    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_gem_adapter_learnable_p_has_parameter(x_4d: Tensor):
    adapter = GeMAdapter(p=3.0, learnable_p=True)

    out = adapter(x_4d.clamp_min(0.1))

    assert out.shape == (4, 6)
    assert isinstance(adapter.p, torch.nn.Parameter)


def test_logsumexp_pool_adapter_fixed_temperature(x_4d: Tensor):
    adapter = LogSumExpPoolAdapter(temperature=1.0, learnable=False)

    out = adapter(x_4d)

    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_logsumexp_pool_adapter_learnable_temperature(x_4d: Tensor):
    adapter = LogSumExpPoolAdapter(temperature=1.0, learnable=True)

    out = adapter(x_4d)

    assert out.shape == (4, 6)
    assert isinstance(adapter.temperature, torch.nn.Parameter)


def test_attn_pool_adapter_shared_mode(x_4d: Tensor):
    adapter = AttnPoolAdapter(in_channels=6, score_mode="shared")

    out = adapter(x_4d)

    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_attn_pool_adapter_per_channel_mode(x_4d: Tensor):
    adapter = AttnPoolAdapter(in_channels=6, score_mode="per_channel")

    out = adapter(x_4d)

    assert out.shape == (4, 6)
    assert torch.isfinite(out).all()


def test_attn_pool_adapter_rejects_bad_mode():
    with pytest.raises(ValueError, match="score_mode"):
        AttnPoolAdapter(in_channels=6, score_mode="bad")


def test_spp_adapter_4d_avg(x_4d: Tensor):
    adapter = SPPAdapter(bins=(1, 2), mode="avg")

    out = adapter(x_4d)

    # bins=(1,2), nd=2 => C*(1*1 + 2*2) = 6*(1 + 4) = 30
    assert out.shape == (4, 30)


def test_spp_adapter_5d_max(x_5d: Tensor):
    adapter = SPPAdapter(bins=(1, 2), mode="max")

    out = adapter(x_5d)

    # bins=(1,2), nd=3 => C*(1^3 + 2^3) = 6*(1 + 8) = 54
    assert out.shape == (2, 54)


def test_spp_adapter_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode"):
        SPPAdapter(bins=(1, 2), mode="bad")


def test_spp_adapter_rejects_empty_bins():
    with pytest.raises(ValueError, match="non-empty"):
        SPPAdapter(bins=())


def test_conv1x1_gap_adapter_4d(x_4d: Tensor):
    adapter = Conv1x1GAPAdapter(in_channels=6, out_channels=4)

    out = adapter(x_4d)

    assert out.shape == (4, 4)
    assert torch.isfinite(out).all()


def test_conv1x1_gap_adapter_rejects_nd_gt_3():
    x_6d = torch.randn(2, 6, 2, 3, 4, 5)
    adapter = Conv1x1GAPAdapter(in_channels=6, out_channels=4)

    with pytest.raises(ValueError, match="spatial dims 1..3"):
        adapter(x_6d)


def test_token_adapter_4d(x_4d: Tensor):
    adapter = TokenAdapter(in_channels=6, num_tokens=3)

    out = adapter(x_4d)

    assert out.shape == (4, 18)
    assert torch.isfinite(out).all()


def test_token_adapter_rejects_wrong_channel_count(x_4d: Tensor):
    adapter = TokenAdapter(in_channels=5, num_tokens=3)

    with pytest.raises(ValueError, match="expected C=5"):
        adapter(x_4d)


def test_token_adapter_rejects_nonpositive_num_tokens():
    with pytest.raises(ValueError, match="positive"):
        TokenAdapter(in_channels=6, num_tokens=0)


def test_feature_adapter_factory_builds_gap():
    spec = FeatureAdapterSpec(
        cls=GAPAdapter,
        kwargs={"keepdim": True},
    )

    adapter = FeatureAdapterFactory.build(spec)

    assert isinstance(adapter, GAPAdapter)
    assert adapter.keepdim is True


def test_feature_adapter_factory_rejects_missing_cls():
    spec = FeatureAdapterSpec(cls=None)

    with pytest.raises(ValueError, match="must be specified"):
        FeatureAdapterFactory.build(spec)


def test_feature_adapter_factory_rejects_non_adapter_cls():
    class NotAnAdapter:
        pass

    spec = FeatureAdapterSpec(cls=NotAnAdapter)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a subclass of FeatureAdapter"):
        FeatureAdapterFactory.build(spec)


def test_feature_adapter_factory_rejects_both_state_dict_and_path(tmp_path):
    spec = FeatureAdapterSpec(cls=GAPAdapter, kwargs={})
    path = tmp_path / "dummy.pt"
    torch.save({}, path)

    with pytest.raises(ValueError, match="Only one of state_dict_path or state_dict"):
        FeatureAdapterFactory.build(
            spec,
            state_dict_path=str(path),
            state_dict={},
        )


def test_feature_adapter_factory_can_load_state_dict():
    original = StatefulFeatureAdapter(scale=2.5)
    state_dict = original.state_dict()

    spec = FeatureAdapterSpec(cls=StatefulFeatureAdapter, kwargs={"scale": 1.0})
    loaded = FeatureAdapterFactory.build(spec, state_dict=state_dict)

    assert isinstance(loaded, StatefulFeatureAdapter)
    assert torch.allclose(loaded.scale, torch.tensor(2.5))
# tests/test_feature_adapters.py
from __future__ import annotations

import math
import pytest
import torch
from torch import nn, Tensor

from sktorch.modules.nn.FeatureAdapters import (
    _BaseAdapter,
    AdapterFactory,
    IdentityAdapter,
    FlattenAdapter,
    GAPAdapter,
    GMPAdapter,
    GAPGMPConcatAdapter,
    StatsPoolAdapter,
    GeMAdapter,
    LogSumExpPoolAdapter,
    AttnPoolAdapter,
    TokenAdapter,
    SPPAdapter,
    Conv1x1GAPAdapter,
)


# -----------------------
# helpers
# -----------------------

def _rand(B=4, C=8, *spatial, requires_grad=False) -> Tensor:
    x = torch.randn((B, C, *spatial), dtype=torch.float32)
    x.requires_grad_(requires_grad)
    return x


class NotAnAdapter(nn.Module):
    def __init__(self):
        super().__init__()


# -----------------------
# AdapterFactory tests
# -----------------------

def test_adapter_factory_from_type_sets_cls_path_and_kwargs():
    f = AdapterFactory.from_type(GAPAdapter, keepdim=True)
    assert isinstance(f.cls_path, str)
    assert f.kwargs == {"keepdim": True}


def test_adapter_factory_build_constructs_adapter():
    f = AdapterFactory.from_type(GAPAdapter, keepdim=False)
    a = f.build()
    assert isinstance(a, _BaseAdapter)
    assert isinstance(a, GAPAdapter)
    assert a.keepdim is False


def test_adapter_factory_build_raises_if_not_adapter():
    f = AdapterFactory.from_type(NotAnAdapter)
    with pytest.raises(TypeError):
        _ = f.build()


def test_adapter_factory_to_dict_from_dict_roundtrip():
    f = AdapterFactory.from_type(StatsPoolAdapter, eps=1e-5)
    d = f.to_dict()
    assert d["__type__"] == "AdapterFactory"
    assert "cls_path" in d
    assert d["kwargs"] == {"eps": 1e-5}

    f2 = AdapterFactory.from_dict(d)
    assert f2.cls_path == f.cls_path
    assert f2.kwargs == f.kwargs

    a = f2.build()
    assert isinstance(a, StatsPoolAdapter)
    assert math.isclose(a.eps, 1e-5)


# -----------------------
# Adapter contract tests
# -----------------------

@pytest.mark.parametrize(
    "adapter",
    [
        IdentityAdapter(),
        FlattenAdapter(),
        GAPAdapter(),
        GMPAdapter(),
        GAPGMPConcatAdapter(),
        StatsPoolAdapter(),
        GeMAdapter(learnable_p=False),
        LogSumExpPoolAdapter(learnable=False),
    ],
)
def test_adapters_accept_tensor_and_return_tensor(adapter: _BaseAdapter):
    x = _rand(3, 5, 7, 7)
    y = adapter(x)
    assert isinstance(y, torch.Tensor)


def test_identity_adapter_preserves_shape():
    x = _rand(2, 3, 4, 5)
    y = IdentityAdapter()(x)
    assert y.shape == x.shape


def test_flatten_adapter_outputs_2d():
    x = _rand(2, 3, 4, 5)
    y = FlattenAdapter()(x)
    assert y.ndim == 2
    assert y.shape == (2, 3 * 4 * 5)


def test_gap_adapter_outputs_BC_by_default_and_keepdim_option():
    x = _rand(2, 3, 4, 5)
    y = GAPAdapter(keepdim=False)(x)
    assert y.shape == (2, 3)

    yk = GAPAdapter(keepdim=True)(x)
    assert yk.shape == (2, 3, 1, 1)


def test_gmp_adapter_outputs_BC_by_default_and_keepdim_option():
    x = _rand(2, 3, 4, 5)
    y = GMPAdapter(keepdim=False)(x)
    assert y.shape == (2, 3)

    yk = GMPAdapter(keepdim=True)(x)
    assert yk.shape == (2, 3, 1, 1)


def test_gapgmp_concat_adapter_outputs_2C():
    x = _rand(2, 3, 4, 5)
    y = GAPGMPConcatAdapter()(x)
    assert y.shape == (2, 6)  # 2C


def test_stats_pool_adapter_outputs_2C_and_is_finite():
    x = _rand(2, 3, 4, 5)
    y = StatsPoolAdapter(eps=1e-6)(x)
    assert y.shape == (2, 6)
    assert torch.isfinite(y).all()


def test_gem_adapter_outputs_BC_and_is_finite_even_for_negative_inputs():
    # GeM clamps to eps, so negatives should not produce NaNs
    x = -torch.abs(_rand(2, 3, 4, 5))
    y = GeMAdapter(p=3.0, eps=1e-6, learnable_p=False)(x)
    assert y.shape == (2, 3)
    assert torch.isfinite(y).all()


def test_logsumexp_pool_adapter_outputs_BC_and_is_finite():
    x = _rand(2, 3, 4, 5)
    y = LogSumExpPoolAdapter(temperature=2.0, learnable=False)(x)
    assert y.shape == (2, 3)
    assert torch.isfinite(y).all()


def test_attn_pool_adapter_shared_outputs_BC():
    x = _rand(2, 6, 4, 5)
    y = AttnPoolAdapter(in_channels=6, score_mode="shared")(x)
    assert y.shape == (2, 6)


def test_attn_pool_adapter_per_channel_outputs_BC():
    x = _rand(2, 6, 4, 5)
    y = AttnPoolAdapter(in_channels=6, score_mode="per_channel")(x)
    assert y.shape == (2, 6)


def test_attn_pool_adapter_invalid_mode_raises():
    with pytest.raises(ValueError):
        _ = AttnPoolAdapter(in_channels=4, score_mode="nope")


def test_token_adapter_outputs_B_by_KC():
    x = _rand(2, 8, 4, 5)
    y = TokenAdapter(in_channels=8, num_tokens=3)(x)
    assert y.shape == (2, 3 * 8)


def test_token_adapter_requires_ndim_ge_3():
    x = torch.randn(2, 8)  # no spatial dims
    with pytest.raises(ValueError):
        _ = TokenAdapter(in_channels=8, num_tokens=2)(x)


def test_token_adapter_requires_matching_channels():
    x = _rand(2, 7, 4, 5)
    with pytest.raises(ValueError):
        _ = TokenAdapter(in_channels=8, num_tokens=2)(x)


def test_spp_adapter_output_dim_2d_avg():
    # for 2D spatial: bins=(1,2) => 1*1 + 2*2 = 5 pooled cells per channel
    x = _rand(2, 3, 8, 8)
    y = SPPAdapter(bins=(1, 2), mode="avg")(x)
    assert y.ndim == 2
    assert y.shape == (2, 3 * (1 * 1 + 2 * 2))


def test_spp_adapter_output_dim_2d_max():
    x = _rand(2, 3, 8, 8)
    y = SPPAdapter(bins=(1, 2), mode="max")(x)
    assert y.shape == (2, 3 * (1 * 1 + 2 * 2))


def test_spp_adapter_invalid_mode_raises():
    with pytest.raises(ValueError):
        _ = SPPAdapter(bins=(1, 2), mode="nope")


def test_spp_adapter_invalid_bins_raise():
    with pytest.raises(ValueError):
        _ = SPPAdapter(bins=(), mode="avg")
    with pytest.raises(ValueError):
        _ = SPPAdapter(bins=(0, 2), mode="avg")


def test_spp_adapter_nd_greater_than_3_raises():
    # spatial_ndim=4 -> adaptive pooling dispatch should raise
    x = _rand(2, 3, 2, 2, 2, 2)  # (B,C,2,2,2,2)
    with pytest.raises(ValueError):
        _ = SPPAdapter(bins=(1, 2), mode="avg")(x)


def test_conv1x1gap_adapter_outputs_out_channels_and_lazy_builds_conv():
    x = _rand(2, 5, 8, 8)
    a = Conv1x1GAPAdapter(in_channels=5, out_channels=7, bias=True)
    assert a._conv is None  # lazy

    y = a(x)
    assert y.shape == (2, 7)
    assert a._conv is not None  # built on first forward


def test_conv1x1gap_adapter_nd_greater_than_3_raises():
    x = _rand(2, 5, 2, 2, 2, 2)  # spatial_ndim=4
    a = Conv1x1GAPAdapter(in_channels=5, out_channels=7)
    with pytest.raises(ValueError):
        _ = a(x)

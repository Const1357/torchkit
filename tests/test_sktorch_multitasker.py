# tests/test_sktorch_multitasker.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Set

import numpy as np
import pytest
import torch
from torch import nn, Tensor

from sktorch.modules.nn.models.multitask import SKTorchMultitasker
from sktorch.modules.nn.models.factory import ModuleFactory
from sktorch.modules.nn.FeatureAdapters import _BaseAdapter, AdapterFactory


# -----------------------
# Local BackboneOut (matches contract used by multitasker)
# -----------------------

@dataclass(frozen=True)
class BackboneOut:
    features: Dict[str, Tensor | None]
    details: Dict[str, Any] = field(default_factory=dict)


# -----------------------
# Dummy backbones
# -----------------------

class DummyBackboneNoGating(nn.Module):
    """
    Ignores requested_features. Always returns all keys in `feature_map`.
    """
    def __init__(self, feature_map: Dict[str, Tensor | None]) -> None:
        super().__init__()
        self.feature_map = feature_map
        self.last_requested_features: Optional[Set[str]] = None

    def forward(self, x: Tensor, **kwargs: Any) -> BackboneOut:
        self.last_requested_features = kwargs.get("requested_features", None)
        return BackboneOut(features=dict(self.feature_map), details={"bb_called": True, "kwargs": dict(kwargs)})


class DummyBackboneWithGating(nn.Module):
    """
    Supports compute gating via requested_features kwarg:
    - If requested_features is provided, returns only those keys.
    - Otherwise returns everything in base_feature_map.
    Records last_requested_features for assertions.
    """
    def __init__(self, *, base_feature_map: Mapping[str, Tensor]):
        super().__init__()
        self.base_feature_map = dict(base_feature_map)
        self.last_requested_features: Optional[Set[str]] = None
        self.calls: int = 0

    def forward(
        self,
        x: Tensor,
        *,
        requested_features: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> BackboneOut:
        self.calls += 1
        self.last_requested_features = None if requested_features is None else set(requested_features)

        if requested_features is None:
            feats: Dict[str, Tensor | None] = {k: v for k, v in self.base_feature_map.items()}
        else:
            feats = {k: self.base_feature_map.get(k, None) for k in requested_features}

        return BackboneOut(features=feats, details={"kwargs": dict(kwargs)})


class DummyBackboneBadFeaturesNotDict(nn.Module):
    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        return type("BackboneOut", (), {"features": x, "details": {}})()  # features is Tensor, not dict


class DummyBackboneBadFeatureNonTensor(nn.Module):
    def forward(self, x: Tensor, **kwargs: Any) -> BackboneOut:
        return BackboneOut(features={"f1": "not a tensor"}, details={})


class DummyBackboneBadFeature1D(nn.Module):
    def forward(self, x: Tensor, **kwargs: Any) -> BackboneOut:
        # (B,) => ndim=1 should raise
        return BackboneOut(features={"f1": torch.zeros(x.shape[0])}, details={})


class DummyBackboneNotAModule:
    def __init__(self, **kwargs: Any) -> None:
        pass


# -----------------------
# Dummy adapters
# -----------------------

class DummyAdapter(nn.Module):
    """
    Minimal adapter-like module: returns input as-is, and records kwargs.
    Must satisfy `_BaseAdapter` type in real code; we will use IdentityAdapter via AdapterFactory
    for most tests. This is only for cases where we want adapter behavior variation.

    NOTE: We intentionally do NOT use this in AdapterFactory.from_type, because AdapterFactory.build()
    checks isinstance(adapter, _BaseAdapter). So we rely on real adapters from FeatureAdapters module.
    """
    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        return x


# -----------------------
# Dummy heads
# -----------------------

@dataclass(frozen=True)
class HeadOut:
    y: Tensor
    details: Dict[str, Any] = field(default_factory=dict)


class DummyHead(nn.Module):
    """
    Task head that accepts input_shape in __init__, then outputs a linear projection.
    """
    def __init__(self, *, input_shape: tuple[int, ...], out_dim: int = 2, tag: str = "") -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.tag = str(tag)
        self.linear = nn.Linear(int(np.prod(input_shape)), self.out_dim)

    def forward(self, x: Tensor, **kwargs: Any) -> HeadOut:
        x2 = x.view(x.shape[0], -1)
        y = self.linear(x2)
        return HeadOut(y=y, details={"head_tag": self.tag, "kwargs": dict(kwargs)})


class DummyHeadNoDetails(nn.Module):
    """
    Returns an object without `.details`; multitasker must NOT add to heads_details.
    """
    def __init__(self, *, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.linear = nn.Linear(int(np.prod(input_shape)), 1)

    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        y = self.linear(x.view(x.shape[0], -1))
        return type("NoDetailsOut", (), {"y": y})()


class DummyHeadBadDetailsType(nn.Module):
    """
    Returns `.details` but not a dict; multitasker must ignore it.
    """
    def __init__(self, *, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        self.linear = nn.Linear(int(np.prod(input_shape)), 1)

    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        y = self.linear(x.view(x.shape[0], -1))
        return type("BadDetailsOut", (), {"y": y, "details": ["not", "a", "dict"]})()


class DummyHeadNotAModule:
    def __init__(self, *, input_shape: tuple[int, ...], **kwargs: Any) -> None:
        pass


# -----------------------
# Helpers
# -----------------------

def _make_mt(
    *,
    backbone_factory: ModuleFactory,
    head_factories: Mapping[str, ModuleFactory],
    backbone_feature_for_task: Mapping[str, str],
    adapter_factories: Mapping[str, AdapterFactory] | None = None,
) -> SKTorchMultitasker:
    return SKTorchMultitasker(
        backbone_factory=backbone_factory,
        head_factories=head_factories,
        adapter_factories=adapter_factories,
        backbone_feature_for_task=backbone_feature_for_task,
        device="cpu",
        dtype=torch.float32,
    )


def _identity_adapter_factories(tasks: list[str]) -> Dict[str, AdapterFactory]:
    return {
        t: AdapterFactory(cls_path="sktorch.modules.nn.FeatureAdapters:IdentityAdapter")
        for t in tasks
    }


# -----------------------
# Tests: init validation
# -----------------------

def test_init_requires_non_empty_head_factories():
    with pytest.raises(ValueError):
        _ = SKTorchMultitasker(
            backbone_factory=ModuleFactory.from_type(DummyBackboneNoGating, feature_map={}),
            head_factories={},
            backbone_feature_for_task={},
            device="cpu",
        )


def test_init_rejects_mismatched_adapter_factories_keys():
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map={"f1": torch.randn(2, 3)})
    heads = {"t1": ModuleFactory.from_type(DummyHead, out_dim=2, tag="t1")}
    adapters = {"OTHER": AdapterFactory(cls_path="sktorch.modules.nn.FeatureAdapters:IdentityAdapter")}
    bfft = {"t1": "f1"}
    with pytest.raises(ValueError, match="adapter_factories keys"):
        _ = _make_mt(backbone_factory=bb, head_factories=heads, adapter_factories=adapters, backbone_feature_for_task=bfft)


def test_init_rejects_mismatched_backbone_feature_for_task_keys():
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map={"f1": torch.randn(2, 3)})
    heads = {"t1": ModuleFactory.from_type(DummyHead, out_dim=2, tag="t1")}
    bfft = {"OTHER": "f1"}
    with pytest.raises(ValueError, match="backbone_feature_for_task keys"):
        _ = _make_mt(backbone_factory=bb, head_factories=heads, adapter_factories=None, backbone_feature_for_task=bfft)


# -----------------------
# Tests: task normalization / unknown tasks
# -----------------------

def test_unknown_active_task_raises_keyerror():
    feature_map = {"f1": torch.randn(2, 5)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {"t1": ModuleFactory.from_type(DummyHead, out_dim=2, tag="t1")}
    bfft = {"t1": "f1"}

    m = _make_mt(backbone_factory=bb, head_factories=heads, backbone_feature_for_task=bfft)
    x = torch.randn(2, 5)

    with pytest.raises(KeyError, match="Unknown task"):
        _ = m(x, active_tasks=["t1", "nope"])


def test_active_tasks_order_is_stable_and_follows_head_factories_order():
    feature_map = {"fa": torch.randn(2, 4), "fb": torch.randn(2, 4)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {
        "taskA": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "taskB": ModuleFactory.from_type(DummyHead, out_dim=3, tag="B"),
    }
    bfft = {"taskA": "fa", "taskB": "fb"}

    m = _make_mt(backbone_factory=bb, head_factories=heads, backbone_feature_for_task=bfft)
    x = torch.randn(2, 4)

    out = m(x, active_tasks=["taskB", "taskA"])  # shuffled input order
    assert list(out.heads_out.keys()) == ["taskA", "taskB"]  # stable mapping order


# -----------------------
# Tests: compute gating behavior
# -----------------------

def test_backbone_gating_requests_only_needed_feature_keys_for_active_tasks():
    base = {
        "fa": torch.randn(2, 6),
        "fb": torch.randn(2, 6),
        "fc": torch.randn(2, 6),
    }
    bb = ModuleFactory.from_type(DummyBackboneWithGating, base_feature_map=base)

    heads = {
        "taskA": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "taskB": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    bfft = {"taskA": "fa", "taskB": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["taskA", "taskB"]),
    )

    x = torch.randn(2, 6)
    _ = m(x, active_tasks=["taskB"])

    assert m.backbone is not None
    assert isinstance(m.backbone, DummyBackboneWithGating)
    assert m.backbone.last_requested_features == {"fb"}


def test_backbone_without_gating_does_not_receive_requested_features_kwarg():
    feature_map = {"fa": torch.randn(2, 6), "fb": torch.randn(2, 6)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {"taskA": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A")}
    bfft = {"taskA": "fa"}

    m = _make_mt(backbone_factory=bb, head_factories=heads, backbone_feature_for_task=bfft)
    x = torch.randn(2, 6)
    _ = m(x, active_tasks=["taskA"])

    assert m.backbone is not None
    assert isinstance(m.backbone, DummyBackboneNoGating)
    assert m.backbone.last_requested_features is None


# -----------------------
# Tests: routing / missing features / type checks
# -----------------------

class DummyBackboneTensor(nn.Module):
    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        return type("BackboneOut", (), {"features": x, "details": {"bb": "tensor"}})()


class DummyBackboneDict(nn.Module):
    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        feats = {"fa": x, "fb": x * 0.0 + 1.0}  # two distinct tensors
        return type("BackboneOut", (), {"features": feats, "details": {"bb": "dict"}})()


class DummyBackboneBadFeatures(nn.Module):
    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        return type("BackboneOut", (), {"features": ["nope"], "details": {}})()


# --- tests ---

def test_backbone_features_tensor_broadcasts_to_multiple_heads_and_respects_active_tasks():
    bb = ModuleFactory.from_type(DummyBackboneTensor)

    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    # mapping is irrelevant for tensor-backbones, but still required by constructor
    bfft = {"A": "fa", "B": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["A", "B"]),
    )

    x = torch.randn(2, 3)

    # tensor backbone broadcasts: both heads run when active_tasks=None
    out_all = m(x)
    assert list(out_all.heads_out.keys()) == ["A", "B"]

    # respects active_tasks: only B runs
    out_b = m(x, active_tasks=["B"])
    assert list(out_b.heads_out.keys()) == ["B"]


def test_backbone_features_dict_routes_by_backbone_feature_for_task_keys():
    bb = ModuleFactory.from_type(DummyBackboneDict)

    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    bfft = {"A": "fa", "B": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["A", "B"]),
    )

    x = torch.randn(2, 3)
    out = m(x, active_tasks=["A", "B"])

    assert list(out.heads_out.keys()) == ["A", "B"]
    assert "A" in out.heads_out and "B" in out.heads_out


def test_backbone_features_neither_tensor_nor_dict_raises_typeerror():
    bb = ModuleFactory.from_type(DummyBackboneBadFeatures)

    heads = {"t1": ModuleFactory.from_type(DummyHead, out_dim=2, tag="t1")}
    bfft = {"t1": "f1"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["t1"]),
    )

    x = torch.randn(2, 3)
    with pytest.raises(TypeError, match=r"features must be either a Tensor.*or a dict"):
        _ = m(x)


def test_missing_required_feature_key_raises_keyerror():
    base = {"fa": torch.randn(2, 5)}  # fb missing
    bb = ModuleFactory.from_type(DummyBackboneWithGating, base_feature_map=base)

    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    bfft = {"A": "fa", "B": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["A", "B"]),
    )

    x = torch.randn(2, 5)
    with pytest.raises(KeyError, match="did not provide feature 'fb'"):
        _ = m(x)  # active_tasks=None => both tasks => requests {fa, fb} and fb will be None


def test_required_feature_present_but_none_raises_keyerror():
    feature_map = {"f1": None}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)
    heads = {"t1": ModuleFactory.from_type(DummyHead, out_dim=2, tag="t1")}
    bfft = {"t1": "f1"}
    m = _make_mt(backbone_factory=bb, head_factories=heads, backbone_feature_for_task=bfft)
    x = torch.randn(2, 3)

    with pytest.raises(KeyError, match="did not provide feature 'f1'"):
        _ = m(x)


def test_feature_value_must_be_tensor():
    bb = ModuleFactory.from_type(DummyBackboneBadFeatureNonTensor)
    heads = {"t1": ModuleFactory.from_type(DummyHead, out_dim=2, tag="t1")}
    bfft = {"t1": "f1"}
    m = _make_mt(backbone_factory=bb, head_factories=heads, backbone_feature_for_task=bfft)
    x = torch.randn(2, 3)

    with pytest.raises(TypeError, match="must be a Tensor"):
        _ = m(x)


def test_feature_tensor_must_be_at_least_2d():
    bb = ModuleFactory.from_type(DummyBackboneBadFeature1D)
    heads = {"t1": ModuleFactory.from_type(DummyHead, out_dim=2, tag="t1")}
    bfft = {"t1": "f1"}
    m = _make_mt(backbone_factory=bb, head_factories=heads, backbone_feature_for_task=bfft)
    x = torch.randn(2, 3)

    with pytest.raises(ValueError, match="must be at least 2D"):
        _ = m(x)


# -----------------------
# Tests: adapter/head lazy init + per-task kwargs routing
# -----------------------

def test_lazy_inits_only_active_tasks_adapters_and_heads():
    feature_map = {"fa": torch.randn(2, 5), "fb": torch.randn(2, 5)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=3, tag="B"),
    }
    bfft = {"A": "fa", "B": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["A", "B"]),
    )

    assert m.backbone is None
    assert m.feature_adapters == {}
    assert m.heads == {}

    x = torch.randn(2, 5)
    _ = m(x, active_tasks=["B"])

    assert m.backbone is not None
    assert set(m.feature_adapters.keys()) == {"B"}
    assert set(m.heads.keys()) == {"B"}


def test_routes_head_fwd_args_per_task():
    feature_map = {"fa": torch.randn(2, 5), "fb": torch.randn(2, 5)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    bfft = {"A": "fa", "B": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["A", "B"]),
    )

    x = torch.randn(2, 5)
    out = m(
        x,
        active_tasks=["A", "B"],
        head_fwd_args={"A": {"ha": "x"}, "B": {"hb": "y"}},
    )

    assert out.heads_details["A"]["kwargs"]["ha"] == "x"
    assert out.heads_details["B"]["kwargs"]["hb"] == "y"


def test_adapter_fwd_args_raises_when_adapter_does_not_accept_kwargs():
    feature_map = {"fa": torch.randn(2, 5)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {"A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A")}
    bfft = {"A": "fa"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["A"]),  # IdentityAdapter
    )

    x = torch.randn(2, 5)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _ = m(
            x,
            active_tasks=["A"],
            adapter_fwd_args={"A": {"a": 1}},
        )


def test_heads_details_collected_only_when_details_is_dict():
    feature_map = {"f": torch.randn(2, 5)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {
        "good": ModuleFactory.from_type(DummyHead, out_dim=2, tag="good"),
        "nodetails": ModuleFactory.from_type(DummyHeadNoDetails),
        "baddetails": ModuleFactory.from_type(DummyHeadBadDetailsType),
    }
    bfft = {"good": "f", "nodetails": "f", "baddetails": "f"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["good", "nodetails", "baddetails"]),
    )

    x = torch.randn(2, 5)
    out = m(x)

    assert "good" in out.heads_details
    assert "nodetails" not in out.heads_details
    assert "baddetails" not in out.heads_details


# -----------------------
# Tests: enforce_fitted
# -----------------------

def test_enforce_fitted_raises_when_any_active_task_not_fitted():
    feature_map = {"fa": torch.randn(2, 5), "fb": torch.randn(2, 5)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    bfft = {"A": "fa", "B": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["A", "B"]),
    )

    # mark only A fitted
    m.task_fitted_["A"] = True
    m.task_fitted_["B"] = False

    x = torch.randn(2, 5)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = m(x, active_tasks=["A", "B"], enforce_fitted=True)


def test_enforce_fitted_allows_when_all_active_tasks_fitted():
    feature_map = {"fa": torch.randn(2, 5), "fb": torch.randn(2, 5)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    bfft = {"A": "fa", "B": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["A", "B"]),
    )

    m.task_fitted_["A"] = True
    m.task_fitted_["B"] = True

    x = torch.randn(2, 5)
    out = m(x, active_tasks=["B"], enforce_fitted=True)
    assert list(out.heads_out.keys()) == ["B"]


# -----------------------
# Tests: backbone/head factory bad builds (ModuleFactory-level errors)
# -----------------------

def test_backbone_factory_must_build_nn_module():
    bb = ModuleFactory.from_type(DummyBackboneNotAModule)
    heads = {"t1": ModuleFactory.from_type(DummyHead, out_dim=2, tag="t1")}
    bfft = {"t1": "f1"}
    m = _make_mt(backbone_factory=bb, head_factories=heads, backbone_feature_for_task=bfft)
    x = torch.randn(2, 3)

    with pytest.raises(TypeError, match=r"Built object is not a torch\.nn\.Module"):
        _ = m(x)


def test_head_factory_must_build_nn_module():
    feature_map = {"f1": torch.randn(2, 5)}
    bb = ModuleFactory.from_type(DummyBackboneNoGating, feature_map=feature_map)

    heads = {"t1": ModuleFactory.from_type(DummyHeadNotAModule)}
    bfft = {"t1": "f1"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_identity_adapter_factories(["t1"]),
    )

    x = torch.randn(2, 5)
    with pytest.raises(TypeError, match=r"Built object is not a torch\.nn\.Module"):
        _ = m(x)


# -----------------------
# Additional Tests
# -----------------------



# -----------------------
# Dummy components (MUST be module-scope; ModuleFactory cannot import <locals>)
# -----------------------

class DummyBackboneWithGating(nn.Module):
    """
    Supports requested_features kwarg and records what it received.

    IMPORTANT: __init__ kwarg name is `base_feature_map` because your existing
    tests construct it with:
        ModuleFactory.from_type(DummyBackboneWithGating, base_feature_map=...)
    """
    def __init__(self, *, base_feature_map: Mapping[str, Tensor]):
        super().__init__()
        self.feature_map = dict(base_feature_map)
        self.last_requested_features: Optional[Set[str]] = None
        self.calls: int = 0

    def forward(self, x: Tensor, *, requested_features: Optional[Set[str]] = None, **kwargs: Any) -> BackboneOut:
        self.calls += 1
        self.last_requested_features = None if requested_features is None else set(requested_features)

        if requested_features is None:
            # return everything
            feats: Dict[str, Tensor | None] = {k: v for k, v in self.feature_map.items()}
        else:
            # return only requested keys (omit others)
            feats = {k: self.feature_map[k] for k in requested_features if k in self.feature_map}

        return BackboneOut(features=feats, details={"bb": "with_gating", "kwargs": dict(kwargs)})


class DummyBackboneNoGatingStrict(nn.Module):
    """
    Does NOT declare requested_features. Fails if it ever receives it.
    """
    def __init__(self, *, feature_map: Mapping[str, Tensor]):
        super().__init__()
        self.feature_map = dict(feature_map)
        self.calls: int = 0

    def forward(self, x: Tensor, **kwargs: Any) -> BackboneOut:
        self.calls += 1
        assert "requested_features" not in kwargs, "requested_features must NOT be passed to a non-gating backbone"
        return BackboneOut(features=dict(self.feature_map), details={"bb": "no_gating", "kwargs": dict(kwargs)})


class PassThroughAdapterKwargs(_BaseAdapter):
    """
    Adapter that accepts **kwargs and records them, to test per-task adapter_fwd_args routing.
    """
    def __init__(self):
        super().__init__()
        self.last_kwargs: Dict[str, Any] = {}
        self.calls: int = 0

    def forward(self, features: Tensor, **kwargs: Any) -> Tensor:
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        return features


class DummyHead(nn.Module):
    """
    Head built via ModuleFactory.from_input(dummy) -> expects input_shape.
    Records kwargs and returns an object with `.details` dict.
    """
    def __init__(self, *, input_shape: tuple[int, ...], out_dim: int = 2, tag: str = "X"):
        super().__init__()
        self.tag = str(tag)
        self.out_dim = int(out_dim)
        in_dim = int(np.prod(input_shape))
        self.lin = nn.Linear(in_dim, self.out_dim)
        self.last_kwargs: Dict[str, Any] = {}
        self.calls: int = 0

    def forward(self, x: Tensor, **kwargs: Any) -> Any:
        self.calls += 1
        self.last_kwargs = dict(kwargs)
        y = self.lin(x.view(x.shape[0], -1))
        # return something that has `.details` dict
        return type(
            "HeadOut",
            (),
            {"pred": y, "details": {"tag": self.tag, "kwargs": dict(kwargs)}},
        )()


# NOTE: MUST be module-scope (no <locals>) for ModuleFactory imports.
class DummyBackboneReturnsNone(nn.Module):
    """
    Backbone that returns a feature key with value None, to test the
    "present but None" path.
    """
    def __init__(self, *, key: str = "fb"):
        super().__init__()
        self.key = key

    def forward(self, x: Tensor, **kwargs: Any) -> BackboneOut:
        return BackboneOut(features={self.key: None}, details={})


def _make_mt(
    *,
    backbone_factory: ModuleFactory,
    head_factories: Mapping[str, ModuleFactory],
    backbone_feature_for_task: Mapping[str, str],
    adapter_factories: Optional[Mapping[str, AdapterFactory]] = None,
) -> SKTorchMultitasker:
    return SKTorchMultitasker(
        backbone_factory=backbone_factory,
        head_factories=head_factories,
        adapter_factories=adapter_factories,
        backbone_feature_for_task=backbone_feature_for_task,
        device="cpu",
        dtype=torch.float32,
    )


def _adapter_factories_for(tasks: list[str]) -> Dict[str, AdapterFactory]:
    return {t: AdapterFactory.from_type(PassThroughAdapterKwargs) for t in tasks}



def test_active_tasks_order_is_stable_and_follows_head_factories_insertion_order():
    feature_map = {"fa": torch.randn(2, 5), "fb": torch.randn(2, 5)}
    bb = ModuleFactory.from_type(DummyBackboneNoGatingStrict, feature_map=feature_map)

    # insertion order is A then B
    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    bfft = {"A": "fa", "B": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_adapter_factories_for(["A", "B"]),
    )

    out = m(torch.randn(2, 5), active_tasks=["B", "A"])
    # dict preserves insertion order; must follow head_factories order (A then B)
    assert list(out.heads_out.keys()) == ["A", "B"]


def test_fast_cache_does_not_cross_talk_between_task_sets_across_calls():
    fa = torch.randn(2, 5)
    fb = torch.randn(2, 5)
    feature_map = {"fa": fa, "fb": fb}
    bb = ModuleFactory.from_type(DummyBackboneNoGatingStrict, feature_map=feature_map)

    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    bfft = {"A": "fa", "B": "fb"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_adapter_factories_for(["A", "B"]),
    )

    x = torch.randn(2, 5)

    out_a1 = m(x, active_tasks=["A"])
    out_b1 = m(x, active_tasks=["B"])
    out_ab = m(x, active_tasks=["A", "B"])
    out_a2 = m(x, active_tasks=["A"])

    # Ensure the correct task keys are present each time
    assert list(out_a1.heads_out.keys()) == ["A"]
    assert list(out_b1.heads_out.keys()) == ["B"]
    assert list(out_ab.heads_out.keys()) == ["A", "B"]
    assert list(out_a2.heads_out.keys()) == ["A"]

    # Ensure the B call did not poison the A routing cache (A still works after)
    assert "A" in out_a2.heads_out
    assert "B" not in out_a2.heads_out


def test_two_tasks_can_share_same_backbone_feature_key_and_get_separate_heads():
    f = torch.randn(2, 6)
    feature_map = {"shared": f}
    bb = ModuleFactory.from_type(DummyBackboneNoGatingStrict, feature_map=feature_map)

    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=3, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=4, tag="B"),
    }
    # both tasks consume the same backbone feature key
    bfft = {"A": "shared", "B": "shared"}

    m = _make_mt(
        backbone_factory=bb,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_adapter_factories_for(["A", "B"]),
    )

    out = m(torch.randn(2, 6), active_tasks=["A", "B"])

    assert set(out.heads_out.keys()) == {"A", "B"}
    # heads must be separate module instances
    assert m.heads["A"] is not m.heads["B"]
    # and details should preserve per-task identity
    assert out.heads_details["A"]["tag"] == "A"
    assert out.heads_details["B"]["tag"] == "B"


def test_missing_feature_key_and_present_but_none_are_both_treated_as_not_computed_and_raise_keyerror():
    # Case 1: missing key
    bb1 = ModuleFactory.from_type(DummyBackboneNoGatingStrict, feature_map={"fa": torch.randn(2, 5)})
    heads = {"A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A")}
    bfft_missing = {"A": "fb"}  # backbone does not provide "fb"
    m1 = _make_mt(
        backbone_factory=bb1,
        head_factories=heads,
        backbone_feature_for_task=bfft_missing,
        adapter_factories=_adapter_factories_for(["A"]),
    )
    with pytest.raises(KeyError, match="did not provide feature"):
        _ = m1(torch.randn(2, 5), active_tasks=["A"])

    # Case 2: present but None (module-scope class; no <locals>)
    bb2 = ModuleFactory.from_type(DummyBackboneReturnsNone, key="fb")
    m2 = _make_mt(
        backbone_factory=bb2,
        head_factories=heads,
        backbone_feature_for_task={"A": "fb"},
        adapter_factories=_adapter_factories_for(["A"]),
    )
    with pytest.raises(KeyError, match="did not provide feature"):
        _ = m2(torch.randn(2, 5), active_tasks=["A"])


def test_requested_features_passed_only_when_backbone_supports_gating_and_matches_active_tasks_needs():
    feature_map = {"fa": torch.randn(2, 5), "fb": torch.randn(2, 5), "fc": torch.randn(2, 5)}
    heads = {
        "A": ModuleFactory.from_type(DummyHead, out_dim=2, tag="A"),
        "B": ModuleFactory.from_type(DummyHead, out_dim=2, tag="B"),
    }
    bfft = {"A": "fa", "B": "fb"}

    # --- gating backbone: must receive requested_features={"fa"} when only task A is active
    bb_gating = ModuleFactory.from_type(DummyBackboneWithGating, base_feature_map=feature_map)
    m_g = _make_mt(
        backbone_factory=bb_gating,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_adapter_factories_for(["A", "B"]),
    )
    _ = m_g(torch.randn(2, 5), active_tasks=["A"])
    assert isinstance(m_g.backbone, DummyBackboneWithGating)
    assert m_g.backbone.last_requested_features == {"fa"}

    # --- non-gating backbone: must NOT receive requested_features at all
    bb_no = ModuleFactory.from_type(DummyBackboneNoGatingStrict, feature_map=feature_map)
    m_no = _make_mt(
        backbone_factory=bb_no,
        head_factories=heads,
        backbone_feature_for_task=bfft,
        adapter_factories=_adapter_factories_for(["A", "B"]),
    )
    _ = m_no(torch.randn(2, 5), active_tasks=["A"])
    assert isinstance(m_no.backbone, DummyBackboneNoGatingStrict)
    assert m_no.backbone.calls == 1

# tests/test_module_factory.py
from __future__ import annotations

import pytest
import torch
from torch import nn, Tensor

from sktorch.modules.nn.models.factory import ModuleFactory


# -----------------------
# Dummy classes for tests
# -----------------------

class DummyBuildOK(nn.Module):
    def __init__(self, *, a: int = 1, b: str = "x"):
        super().__init__()
        self.a = a
        self.b = b


class DummyNeedsDummy(nn.Module):
    def __init__(self, *, dummy: Tensor):
        super().__init__()
        # store something to verify we actually got the dummy
        self.input_shape = tuple(dummy.shape)


class DummyNeedsInputShape(nn.Module):
    def __init__(self, *, input_shape: tuple[int, ...]):
        super().__init__()
        self.input_shape = input_shape


class DummyNoAcceptedArgs(nn.Module):
    def __init__(self):
        super().__init__()


class NotAModule:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


# -----------------------
# Tests
# -----------------------

def test_from_type_sets_cls_path_and_kwargs():
    f = ModuleFactory.from_type(DummyBuildOK, a=7, b="hello")
    assert isinstance(f.cls_path, str)
    assert f.kwargs == {"a": 7, "b": "hello"}


def test_build_constructs_module_and_passes_kwargs():
    f = ModuleFactory.from_type(DummyBuildOK, a=3, b="z")
    m = f.build()
    assert isinstance(m, nn.Module)
    assert isinstance(m, DummyBuildOK)
    assert m.a == 3
    assert m.b == "z"


def test_build_allows_runtime_build_args_extend():
    # runtime args may add new keys that are not already present
    f = ModuleFactory.from_type(DummyBuildOK, a=1, b="x")
    m = f.build()  # no runtime args needed for this class
    assert m.a == 1
    assert m.b == "x"


def test_build_disallows_runtime_override_of_stored_kwargs():
    # runtime args must NOT override stored kwargs
    f = ModuleFactory.from_type(DummyBuildOK, a=1, b="x")
    with pytest.raises(TypeError):
        _ = f.build(a=99)


def test_build_raises_if_built_object_not_nn_module():
    f = ModuleFactory.from_type(NotAModule, x=1)
    with pytest.raises(TypeError):
        _ = f.build()


def test_from_input_requires_dummy_at_least_2d():
    f = ModuleFactory.from_type(DummyNeedsInputShape)
    dummy = torch.randn(8)  # 1D
    with pytest.raises(ValueError):
        _ = f.from_input(dummy)


def test_from_input_prefers_dummy_param_when_available():
    f = ModuleFactory.from_type(DummyNeedsDummy)
    dummy = torch.randn(4, 10, 2)  # (B, ...)
    m = f.from_input(dummy)
    assert isinstance(m, DummyNeedsDummy)
    assert m.input_shape == tuple(dummy.shape)  # stored full dummy shape


def test_from_input_uses_input_shape_when_dummy_param_not_available():
    f = ModuleFactory.from_type(DummyNeedsInputShape)
    dummy = torch.randn(4, 10, 2)  # (B, ...)
    m = f.from_input(dummy)
    assert isinstance(m, DummyNeedsInputShape)
    assert m.input_shape == (10, 2)  # excludes batch dimension


def test_from_input_raises_if_neither_dummy_nor_input_shape_supported():
    f = ModuleFactory.from_type(DummyNoAcceptedArgs)
    dummy = torch.randn(4, 10)
    with pytest.raises(ValueError):
        _ = f.from_input(dummy)


def test_to_dict_and_from_dict_roundtrip():
    f = ModuleFactory.from_type(DummyBuildOK, a=5, b="k")
    d = f.to_dict()

    assert d["__type__"] == "ModuleFactory"
    assert "cls_path" in d
    assert d["kwargs"] == {"a": 5, "b": "k"}

    f2 = ModuleFactory.from_dict(d)
    assert isinstance(f2, ModuleFactory)
    assert f2.cls_path == f.cls_path
    assert f2.kwargs == f.kwargs

    # and it should still build
    m = f2.build()
    assert isinstance(m, DummyBuildOK)
    assert m.a == 5
    assert m.b == "k"

# tests/test_base_estimator.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn, Tensor

from sktorch.modules.nn.models._base._estimator import SKTorchEstimatorBase


# -----------------------
# Dummy estimator for tests
# -----------------------

class DummyEstimator(SKTorchEstimatorBase):
    """
    Minimal concrete estimator to test SKTorchEstimatorBase contracts.
    """

    def __init__(
        self,
        *,
        device: str | torch.device | None = "cpu",
        dtype: torch.dtype = torch.float32,
        scale: float = 1.0,  # extra sklearn param to test init param roundtrip
    ):
        self.scale = float(scale)  # sklearn param must be stored exactly
        super().__init__(device=device, dtype=dtype)
        # create a parameter so we can test state_dict roundtrip + device placement
        self.w = nn.Parameter(torch.tensor(2.0, device=self._device, dtype=self.dtype))

    def forward(self, X: Tensor, **kwargs: Any) -> Tensor:
        return X.to(device=self._device, dtype=self.dtype) * self.w * self.scale

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "DummyEstimator":
        # not a real fit; just toggles fitted flag for test purposes
        self.is_fitted_ = True
        return self


class DummyEstimatorWithFittedAttrs(DummyEstimator):
    def __init__(
        self,
        *,
        device: str | torch.device | None = "cpu",
        dtype: torch.dtype = torch.float32,
        scale: float = 1.0,
    ):
        super().__init__(device=device, dtype=dtype, scale=scale)
        self.classes_ = np.array([0, 1, 2], dtype=int)

    def _fitted_state_keys(self) -> tuple[str, ...]:
        return super()._fitted_state_keys() + ("classes_",)


# -----------------------
# Tests
# -----------------------

def test_init_sets_device_dtype_and_is_fitted_default_false():
    m = DummyEstimator(device="cpu", dtype=torch.float32, scale=1.5)
    assert m.is_fitted_ is False
    assert str(m._device) == "cpu"
    assert m.dtype == torch.float32
    assert isinstance(m, nn.Module)


def test_parameters_are_on_resolved_device_and_dtype():
    m = DummyEstimator(device="cpu", dtype=torch.float64)
    assert m.w.device.type == "cpu"
    assert m.w.dtype == torch.float64


def test_to_tensor_converts_numpy_to_tensor_on_device_and_dtype():
    m = DummyEstimator(device="cpu", dtype=torch.float64)
    x_np = np.zeros((2, 3), dtype=np.float32)
    x = m._to_tensor(x_np)

    assert isinstance(x, torch.Tensor)
    assert x.device.type == "cpu"
    assert x.dtype == torch.float64
    assert x.shape == (2, 3)


def test_get_init_params_includes_sklearn_params():
    m = DummyEstimator(device="cpu", dtype=torch.float32, scale=2.0)
    params = m.get_init_params()
    # at minimum, should include the explicit sklearn param we added
    assert params["scale"] == 2.0
    # device is stored as provided (raw), not resolved
    assert params["device"] == "cpu"
    assert params["dtype"] == torch.float32


def test_fitted_state_default_contains_is_fitted_only():
    m = DummyEstimator(device="cpu")
    st = m._get_fitted_state()
    assert st == {"is_fitted_": False}


def test_fitted_state_can_be_extended_and_persists():
    m = DummyEstimatorWithFittedAttrs(device="cpu")
    st = m._get_fitted_state()
    assert "is_fitted_" in st
    assert "classes_" in st
    assert np.array_equal(st["classes_"], np.array([0, 1, 2], dtype=int))


def test_save_load_roundtrip_restores_type_params_state_and_fitted_attrs(tmp_path):
    path = tmp_path / "estimator.pt"

    m = DummyEstimatorWithFittedAttrs(device="cpu", dtype=torch.float32, scale=3.0)
    # mutate parameter deterministically
    with torch.no_grad():
        m.w.copy_(torch.tensor(7.0))

    m.is_fitted_ = True

    m.save(str(path))
    m2 = SKTorchEstimatorBase.load(str(path), map_location="cpu", strict=True)

    assert isinstance(m2, DummyEstimatorWithFittedAttrs)
    assert m2.scale == 3.0
    assert m2.is_fitted_ is True
    assert np.array_equal(m2.classes_, np.array([0, 1, 2], dtype=int))

    # state_dict roundtrip
    assert torch.allclose(m2.w.detach().cpu(), torch.tensor(7.0))


def test_estimator_state_format_and_class_metadata_present():
    m = DummyEstimator(device="cpu")
    st = m.estimator_state()

    assert st["format"]["name"] == "sktorch-estimator"
    assert st["format"]["version"] == 1
    assert "class" in st and "module" in st["class"] and "name" in st["class"]
    assert "init_params" in st
    assert "model_state_dict" in st
    assert "fitted_state" in st


def test_load_raises_on_wrong_format_name(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(
        {
            "format": {"name": "not-sktorch-estimator", "version": 1},
            "class": {"module": "x", "name": "y"},
            "init_params": {},
            "model_state_dict": {},
            "fitted_state": {},
        },
        str(path),
    )

    with pytest.raises(ValueError):
        _ = SKTorchEstimatorBase.load(str(path), map_location="cpu")


def test_import_by_name_rejects_non_class_object():
    # os.path is a module attribute that's not a class
    with pytest.raises(TypeError):
        _ = SKTorchEstimatorBase._import_by_name("os", "path")

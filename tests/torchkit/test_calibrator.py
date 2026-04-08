from __future__ import annotations

import pytest
import torch

from torchkit.models.calibration._calibrator import Calibrator
from torchkit.models.calibration.factory import CalibratorFactory, CalibratorSpec
from torchkit.models.calibration.temperature import TemperatureScalingCalibrator
from torchkit.models.calibration.platt import PlattScalingCalibrator
from torchkit.models.calibration.isotonic import IsotonicRegressionCalibrator


class DummyCalibrator(Calibrator):
    def forward_impl(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + 1.0

    def fit_impl(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        self._fit_called = True


class BadShapeCalibrator(Calibrator):
    def forward_impl(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.unsqueeze(-1)

    def fit_impl(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        pass


@pytest.fixture
def multiclass_logits() -> torch.Tensor:
    return torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [0.1, 1.2, -0.4],
            [-0.3, 0.7, 1.5],
            [1.1, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def multiclass_targets() -> torch.Tensor:
    return torch.tensor([0, 1, 2, 0], dtype=torch.long)


@pytest.fixture
def binary_logits_n() -> torch.Tensor:
    return torch.tensor([2.0, -1.0, 0.5, -0.3, 1.2], dtype=torch.float32)


@pytest.fixture
def binary_logits_n1() -> torch.Tensor:
    return torch.tensor([[2.0], [-1.0], [0.5], [-0.3], [1.2]], dtype=torch.float32)


@pytest.fixture
def binary_logits_n2() -> torch.Tensor:
    return torch.tensor(
        [
            [-1.0, 1.0],
            [1.5, -0.5],
            [-0.2, 0.2],
            [0.3, -0.3],
            [-1.2, 1.2],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def binary_targets() -> torch.Tensor:
    return torch.tensor([1, 0, 1, 0, 1], dtype=torch.long)


def test_base_calibrator_is_inactive_by_default():
    cal = DummyCalibrator()
    assert cal.is_active is False


def test_base_calibrator_enable_disable():
    cal = DummyCalibrator()
    assert cal.is_active is False

    returned = cal.enable()
    assert returned is cal
    assert cal.is_active is True

    returned = cal.disable()
    assert returned is cal
    assert cal.is_active is False


def test_base_calibrator_forward_skips_when_inactive(binary_logits_n2: torch.Tensor):
    cal = DummyCalibrator(active=False)
    out = cal(binary_logits_n2)
    assert torch.equal(out, binary_logits_n2)


def test_base_calibrator_forward_applies_when_active(binary_logits_n2: torch.Tensor):
    cal = DummyCalibrator(active=True)
    out = cal(binary_logits_n2)
    assert torch.allclose(out, binary_logits_n2 + 1.0)


def test_base_calibrator_forward_checks_output_shape(binary_logits_n2: torch.Tensor):
    cal = BadShapeCalibrator(active=True)
    with pytest.raises(ValueError, match="same shape"):
        _ = cal(binary_logits_n2)


def test_base_calibrator_fit_checks_batch_size(binary_logits_n2: torch.Tensor, binary_targets: torch.Tensor):
    cal = DummyCalibrator()
    with pytest.raises(ValueError, match="matching batch size"):
        cal.fit(binary_logits_n2, binary_targets[:-1])


def test_base_calibrator_fit_checks_tensor_types(binary_logits_n2: torch.Tensor, binary_targets: torch.Tensor):
    cal = DummyCalibrator()

    with pytest.raises(ValueError, match="`logits` to be a Tensor"):
        cal.fit([1, 2, 3], binary_targets)

    with pytest.raises(ValueError, match="`targets` to be a Tensor"):
        cal.fit(binary_logits_n2, [1, 0, 1])


def test_temperature_scaling_forward_preserves_shape(multiclass_logits: torch.Tensor):
    cal = TemperatureScalingCalibrator(init_temp=2.0, active=True)
    out = cal(multiclass_logits)

    assert out.shape == multiclass_logits.shape
    assert torch.allclose(out, multiclass_logits / 2.0)


def test_temperature_scaling_fit_updates_or_keeps_valid_temperature(
    multiclass_logits: torch.Tensor,
    multiclass_targets: torch.Tensor,
):
    cal = TemperatureScalingCalibrator(init_temp=1.5, max_iter=5, lr=0.1, active=True)
    before = cal.temperature.detach().clone()

    cal.fit(multiclass_logits, multiclass_targets)

    after = cal.temperature.detach().clone()
    assert after.numel() == 1
    assert float(after.item()) > 0.0
    assert torch.isfinite(after).all()
    assert before.shape == after.shape


def test_temperature_scaling_rejects_non_multiclass(binary_logits_n1: torch.Tensor, binary_targets: torch.Tensor):
    cal = TemperatureScalingCalibrator(active=True)

    with pytest.raises(ValueError):
        cal.fit(binary_logits_n1, binary_targets)


@pytest.mark.parametrize("logits_fixture", ["binary_logits_n", "binary_logits_n1", "binary_logits_n2"])
def test_platt_scaling_forward_preserves_input_shape(
    request: pytest.FixtureRequest,
    logits_fixture: str,
):
    logits = request.getfixturevalue(logits_fixture)
    cal = PlattScalingCalibrator(init_a=1.2, init_b=-0.1, active=True)

    out = cal(logits)
    assert out.shape == logits.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("logits_fixture", ["binary_logits_n", "binary_logits_n1", "binary_logits_n2"])
def test_platt_scaling_fit_runs_for_supported_binary_shapes(
    request: pytest.FixtureRequest,
    logits_fixture: str,
    binary_targets: torch.Tensor,
):
    logits = request.getfixturevalue(logits_fixture)
    cal = PlattScalingCalibrator(max_iter=5, lr=0.1, active=True)

    cal.fit(logits, binary_targets)

    assert cal.a.numel() == 1
    assert cal.b.numel() == 1
    assert float(cal.a.item()) > 0.0
    assert torch.isfinite(cal.a).all()
    assert torch.isfinite(cal.b).all()


def test_platt_scaling_forward_remains_monotone_when_parameter_is_negative(binary_logits_n: torch.Tensor):
    cal = PlattScalingCalibrator(active=True)
    with torch.no_grad():
        cal.a.fill_(-3.0)
        cal.b.fill_(0.25)

    out = cal(binary_logits_n)
    sorted_in = torch.argsort(binary_logits_n)
    sorted_out = out[sorted_in]

    assert out.shape == binary_logits_n.shape
    assert torch.isfinite(out).all()
    assert torch.all(torch.diff(sorted_out) >= 0.0)


def test_platt_scaling_rejects_non_binary_targets(binary_logits_n2: torch.Tensor):
    cal = PlattScalingCalibrator(active=True)
    bad_targets = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="binary targets"):
        cal.fit(binary_logits_n2, bad_targets)


@pytest.mark.parametrize("logits_fixture", ["binary_logits_n", "binary_logits_n1", "binary_logits_n2"])
def test_isotonic_forward_requires_fit(
    request: pytest.FixtureRequest,
    logits_fixture: str,
):
    logits = request.getfixturevalue(logits_fixture)
    cal = IsotonicRegressionCalibrator(active=True)

    with pytest.raises(ValueError, match="must be fit before calling forward"):
        _ = cal(logits)


@pytest.mark.parametrize("logits_fixture", ["binary_logits_n", "binary_logits_n1", "binary_logits_n2"])
def test_isotonic_fit_and_forward_supported_binary_shapes(
    request: pytest.FixtureRequest,
    logits_fixture: str,
    binary_targets: torch.Tensor,
):
    logits = request.getfixturevalue(logits_fixture)
    cal = IsotonicRegressionCalibrator(active=True)

    cal.fit(logits, binary_targets)
    out = cal(logits)

    assert cal.is_fitted is True
    assert out.shape == logits.shape
    assert torch.isfinite(out).all()


def test_isotonic_rejects_non_binary_targets(binary_logits_n: torch.Tensor):
    cal = IsotonicRegressionCalibrator(active=True)
    bad_targets = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="binary targets"):
        cal.fit(binary_logits_n, bad_targets)


def test_factory_builds_calibrator_instance():
    spec = CalibratorSpec(
        cls=TemperatureScalingCalibrator,
        kwargs={"init_temp": 2.0},
        active=True,
    )

    cal = CalibratorFactory.build(spec)

    assert isinstance(cal, TemperatureScalingCalibrator)
    assert cal.is_active is True
    assert pytest.approx(float(cal.temperature.item())) == 2.0

def test_factory_respects_inactive_spec():
    spec = CalibratorSpec(
        cls=TemperatureScalingCalibrator,
        kwargs={"init_temp": 2.0},
        active=False,
    )

    cal = CalibratorFactory.build(spec)

    assert isinstance(cal, TemperatureScalingCalibrator)
    assert cal.is_active is False


def test_factory_rejects_missing_cls():
    spec = CalibratorSpec(cls=None)

    with pytest.raises(ValueError, match="must be specified"):
        CalibratorFactory.build(spec)


def test_factory_rejects_both_state_dict_and_path(tmp_path):
    spec = CalibratorSpec(cls=TemperatureScalingCalibrator)
    path = tmp_path / "dummy.pt"
    torch.save({}, path)

    with pytest.raises(ValueError, match="Only one of state_dict_path or state_dict"):
        CalibratorFactory.build(
            spec,
            state_dict_path=str(path),
            state_dict={},
        )


def test_factory_can_load_state_dict_for_temperature_scaling():
    original = TemperatureScalingCalibrator(init_temp=3.0, active=False)
    state_dict = original.state_dict()

    spec = CalibratorSpec(cls=TemperatureScalingCalibrator, kwargs={"init_temp": 1.0})
    loaded = CalibratorFactory.build(spec, state_dict=state_dict)

    assert isinstance(loaded, TemperatureScalingCalibrator)
    assert pytest.approx(float(loaded.temperature.item())) == 3.0

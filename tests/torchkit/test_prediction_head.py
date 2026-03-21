from __future__ import annotations

import pytest
import torch

from torchkit.models.prediction._prediction_head import PredictionHead
from torchkit.models.prediction.factory import PredictionHeadFactory, PredictionHeadSpec

from torchkit.models.calibration._calibrator import Calibrator
from torchkit.models.calibration.factory import CalibratorSpec

from torchkit.models.probability_mapping._probability_mapper import ProbabilityMapper
from torchkit.models.probability_mapping.factory import ProbabilityMapperSpec

from torchkit.models.decision._decision_module import DecisionModule
from torchkit.models.decision.factory import DecisionModuleSpec


class DummyCalibrator(Calibrator):
    def forward_impl(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + 1.0

    def fit_impl(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        pass


class DummyProbabilityMapper(ProbabilityMapper):
    def forward_impl(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1:
            return torch.sigmoid(logits)
        if logits.ndim == 2 and logits.shape[1] == 1:
            return torch.sigmoid(logits)
        if logits.ndim == 2:
            return torch.softmax(logits, dim=1)
        raise ValueError("Unsupported logits shape.")


class DummyDecisionModule(DecisionModule):
    def forward_impl(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim == 1:
            return (probs >= 0.5).long()
        if probs.ndim == 2 and probs.shape[1] == 1:
            return (probs[:, 0] >= 0.5).long()
        if probs.ndim == 2:
            return torch.argmax(probs, dim=1)
        raise ValueError("Unsupported probabilities shape.")


class TrainableDummyDecisionModule(DummyDecisionModule):
    def fit_impl(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        return None


@pytest.fixture
def head_out_binary_n2() -> dict[str, torch.Tensor]:
    return {
        "logits": torch.tensor(
            [
                [-1.0, 1.0],
                [1.5, -0.5],
                [-0.2, 0.2],
            ],
            dtype=torch.float32,
        ),
        "aux": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    }


@pytest.fixture
def head_out_binary_n1() -> dict[str, torch.Tensor]:
    return {
        "logits": torch.tensor([[2.0], [-1.0], [0.5]], dtype=torch.float32),
        "aux": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
    }


def test_prediction_head_is_active_by_default():
    head = PredictionHead()
    assert head.is_active is True


def test_prediction_head_enable_disable():
    head = PredictionHead(active=False)
    assert head.is_active is False

    returned = head.enable()
    assert returned is head
    assert head.is_active is True

    returned = head.disable()
    assert returned is head
    assert head.is_active is False


def test_prediction_head_has_active_calibrator_property():
    head = PredictionHead(calibrator=DummyCalibrator(active=False))
    assert head.has_active_calibrator is False

    head.calibrator.enable()
    assert head.has_active_calibrator is True


def test_prediction_head_has_trainable_decision_module_property():
    head = PredictionHead(decision_module=DummyDecisionModule())
    assert head.has_trainable_decision_module is False

    head.decision_module = TrainableDummyDecisionModule()
    assert head.has_trainable_decision_module is True


def test_prediction_head_returns_none_when_inactive(head_out_binary_n2: dict[str, torch.Tensor]):
    head = PredictionHead(active=False)
    out = head(head_out_binary_n2)

    assert out is None


def test_prediction_head_requires_dict_input():
    head = PredictionHead()

    with pytest.raises(TypeError, match="expected `head_out` to be a dict"):
        head(torch.tensor([1.0]))


def test_prediction_head_requires_logits_key():
    head = PredictionHead()

    with pytest.raises(KeyError, match="contain 'logits'"):
        head({"aux": torch.tensor([1.0])})


def test_prediction_head_requires_logits_tensor():
    head = PredictionHead()

    with pytest.raises(TypeError, match="head_out\\['logits'\\] to be a Tensor"):
        head({"logits": [1.0, 2.0]})


def test_prediction_head_without_components_returns_raw_head_out(head_out_binary_n2: dict[str, torch.Tensor]):
    head = PredictionHead()

    out = head(head_out_binary_n2)

    assert isinstance(out, dict)
    assert set(out.keys()) == {"logits", "aux"}
    assert torch.equal(out["logits"], head_out_binary_n2["logits"])
    assert torch.equal(out["aux"], head_out_binary_n2["aux"])


def test_prediction_head_with_inactive_calibrator_does_not_add_calibrated_logits(
    head_out_binary_n2: dict[str, torch.Tensor],
):
    head = PredictionHead(
        calibrator=DummyCalibrator(active=False),
        probability_mapper=DummyProbabilityMapper(),
    )

    out = head(head_out_binary_n2)

    assert "calibrated_logits" not in out
    assert "probabilities" in out
    expected_probs = torch.softmax(head_out_binary_n2["logits"], dim=1)
    assert torch.allclose(out["probabilities"], expected_probs)


def test_prediction_head_with_active_calibrator_adds_calibrated_logits(
    head_out_binary_n2: dict[str, torch.Tensor],
):
    head = PredictionHead(
        calibrator=DummyCalibrator(active=True),
        probability_mapper=DummyProbabilityMapper(),
    )

    out = head(head_out_binary_n2)

    assert "calibrated_logits" in out
    assert torch.allclose(out["calibrated_logits"], head_out_binary_n2["logits"] + 1.0)

    expected_probs = torch.softmax(head_out_binary_n2["logits"] + 1.0, dim=1)
    assert "probabilities" in out
    assert torch.allclose(out["probabilities"], expected_probs)


def test_prediction_head_with_decision_module_requires_probability_mapper(
    head_out_binary_n2: dict[str, torch.Tensor],
):
    head = PredictionHead(
        decision_module=DummyDecisionModule(),
    )

    with pytest.raises(RuntimeError, match="cannot apply decision_module without probabilities"):
        head(head_out_binary_n2)


def test_prediction_head_with_probability_mapper_and_decision_module(
    head_out_binary_n2: dict[str, torch.Tensor],
):
    head = PredictionHead(
        probability_mapper=DummyProbabilityMapper(),
        decision_module=DummyDecisionModule(),
    )

    out = head(head_out_binary_n2)

    assert "probabilities" in out
    assert "predictions" in out
    assert out["predictions"].shape == (head_out_binary_n2["logits"].shape[0],)
    assert out["predictions"].dtype == torch.long


def test_prediction_head_preserves_original_head_outputs(
    head_out_binary_n2: dict[str, torch.Tensor],
):
    head = PredictionHead(
        calibrator=DummyCalibrator(active=True),
        probability_mapper=DummyProbabilityMapper(),
        decision_module=DummyDecisionModule(),
    )

    out = head(head_out_binary_n2)

    assert "aux" in out
    assert torch.equal(out["aux"], head_out_binary_n2["aux"])
    assert torch.equal(out["logits"], head_out_binary_n2["logits"])


def test_prediction_head_binary_n1_pipeline(head_out_binary_n1: dict[str, torch.Tensor]):
    head = PredictionHead(
        calibrator=DummyCalibrator(active=True),
        probability_mapper=DummyProbabilityMapper(),
        decision_module=DummyDecisionModule(),
    )

    out = head(head_out_binary_n1)

    assert out["logits"].shape == (3, 1)
    assert out["calibrated_logits"].shape == (3, 1)
    assert out["probabilities"].shape == (3, 1)
    assert out["predictions"].shape == (3,)
    assert out["predictions"].dtype == torch.long


def test_prediction_head_factory_builds_full_prediction_head():
    spec = PredictionHeadSpec(
        calibrator=CalibratorSpec(cls=DummyCalibrator, kwargs={}, active=True),
        probability_mapper=ProbabilityMapperSpec(cls=DummyProbabilityMapper, kwargs={}),
        decision_module=DecisionModuleSpec(cls=DummyDecisionModule, kwargs={}),
        active=True,
    )

    head = PredictionHeadFactory.build(spec)

    assert isinstance(head, PredictionHead)
    assert isinstance(head.calibrator, DummyCalibrator)
    assert isinstance(head.probability_mapper, DummyProbabilityMapper)
    assert isinstance(head.decision_module, DummyDecisionModule)
    assert head.is_active is True
    assert head.has_active_calibrator is True


def test_prediction_head_factory_builds_inactive_prediction_head():
    spec = PredictionHeadSpec(
        calibrator=None,
        probability_mapper=None,
        decision_module=None,
        active=False,
    )

    head = PredictionHeadFactory.build(spec)

    assert isinstance(head, PredictionHead)
    assert head.is_active is False


def test_prediction_head_factory_rejects_both_state_dict_and_path(tmp_path):
    spec = PredictionHeadSpec()
    path = tmp_path / "dummy.pt"
    torch.save({}, path)

    with pytest.raises(ValueError, match="Only one of state_dict_path or state_dict"):
        PredictionHeadFactory.build(
            spec,
            state_dict_path=str(path),
            state_dict={},
        )


def test_prediction_head_factory_rejects_mixing_whole_and_nested_state_loading(tmp_path):
    spec = PredictionHeadSpec(
        calibrator=CalibratorSpec(cls=DummyCalibrator, kwargs={}, active=True),
    )
    path = tmp_path / "dummy.pt"
    torch.save({}, path)

    with pytest.raises(ValueError, match="cannot be mixed with nested component state loading"):
        PredictionHeadFactory.build(
            spec,
            state_dict_path=str(path),
            calibrator_state_dict={},
        )


def test_prediction_head_factory_can_load_whole_state_dict(head_out_binary_n2: dict[str, torch.Tensor]):
    original = PredictionHead(
        calibrator=DummyCalibrator(active=True),
        probability_mapper=DummyProbabilityMapper(),
        decision_module=DummyDecisionModule(),
        active=True,
    )
    state_dict = original.state_dict()

    spec = PredictionHeadSpec(
        calibrator=CalibratorSpec(cls=DummyCalibrator, kwargs={}, active=True),
        probability_mapper=ProbabilityMapperSpec(cls=DummyProbabilityMapper, kwargs={}),
        decision_module=DecisionModuleSpec(cls=DummyDecisionModule, kwargs={}),
        active=True,
    )

    loaded = PredictionHeadFactory.build(spec, state_dict=state_dict)

    out = loaded(head_out_binary_n2)
    assert isinstance(loaded, PredictionHead)
    assert "probabilities" in out
    assert "predictions" in out

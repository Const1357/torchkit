from __future__ import annotations

import pytest
import torch

from torchkit.models.decision._decision_module import DecisionModule
from torchkit.models.decision.classification import (
    BinaryClassificationThreshold,
    ArgmaxDecision,
    SampleTopKTemperature,
)
from torchkit.models.decision.factory import (
    DecisionModuleFactory,
    DecisionModuleSpec,
)


class DummyDecisionModule(DecisionModule):
    def forward_impl(self, probs: torch.Tensor) -> torch.Tensor:
        return probs.long()


class BadTypeDecisionModule(DecisionModule):
    def forward_impl(self, probs: torch.Tensor):
        return {"not": "a tensor"}


@pytest.fixture
def binary_probs_n() -> torch.Tensor:
    return torch.tensor([0.9, 0.2, 0.5, 0.49, 0.51], dtype=torch.float32)


@pytest.fixture
def binary_probs_n1() -> torch.Tensor:
    return torch.tensor([[0.9], [0.2], [0.5], [0.49], [0.51]], dtype=torch.float32)


@pytest.fixture
def binary_probs_n2() -> torch.Tensor:
    return torch.tensor(
        [
            [0.1, 0.9],
            [0.8, 0.2],
            [0.5, 0.5],
            [0.51, 0.49],
            [0.49, 0.51],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def multiclass_probs() -> torch.Tensor:
    return torch.tensor(
        [
            [0.1, 0.7, 0.2],
            [0.8, 0.1, 0.1],
            [0.2, 0.3, 0.5],
            [0.25, 0.25, 0.50],
        ],
        dtype=torch.float32,
    )


def test_base_decision_module_checks_input_type():
    module = DummyDecisionModule()

    with pytest.raises(ValueError, match="expects `probs` to be a Tensor"):
        module([1, 2, 3])


def test_base_decision_module_checks_output_type(binary_probs_n: torch.Tensor):
    module = BadTypeDecisionModule()

    with pytest.raises(ValueError, match="output of `forward_impl` to be a Tensor"):
        module(binary_probs_n)


def test_base_decision_module_passes_valid_tensor_output(binary_probs_n: torch.Tensor):
    module = DummyDecisionModule()
    out = module(binary_probs_n)

    assert isinstance(out, torch.Tensor)
    assert out.shape == binary_probs_n.shape
    assert out.dtype == torch.long


def test_binary_threshold_rejects_invalid_init_threshold():
    with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
        BinaryClassificationThreshold(threshold=-0.1)

    with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
        BinaryClassificationThreshold(threshold=1.1)


def test_binary_threshold_property_setter_validates():
    module = BinaryClassificationThreshold(threshold=0.5)

    module.threshold = 0.7
    assert module.threshold == 0.7

    with pytest.raises(ValueError, match="threshold must be in \\[0, 1\\]"):
        module.threshold = 2.0


@pytest.mark.parametrize("fixture_name", ["binary_probs_n", "binary_probs_n1", "binary_probs_n2"])
def test_binary_threshold_supports_all_binary_shapes(
    request: pytest.FixtureRequest,
    fixture_name: str,
):
    probs = request.getfixturevalue(fixture_name)
    module = BinaryClassificationThreshold(threshold=0.5)

    preds = module(probs)

    assert isinstance(preds, torch.Tensor)
    assert preds.ndim == 1
    assert preds.shape[0] == probs.shape[0]
    assert preds.dtype == torch.long
    assert set(preds.tolist()).issubset({0, 1})


def test_binary_threshold_expected_predictions_for_n_shape(binary_probs_n: torch.Tensor):
    module = BinaryClassificationThreshold(threshold=0.5)
    preds = module(binary_probs_n)

    assert torch.equal(preds, torch.tensor([1, 0, 1, 0, 1], dtype=torch.long))


def test_binary_threshold_expected_predictions_for_n1_shape(binary_probs_n1: torch.Tensor):
    module = BinaryClassificationThreshold(threshold=0.5)
    preds = module(binary_probs_n1)

    assert torch.equal(preds, torch.tensor([1, 0, 1, 0, 1], dtype=torch.long))


def test_binary_threshold_expected_predictions_for_n2_shape(binary_probs_n2: torch.Tensor):
    module = BinaryClassificationThreshold(threshold=0.5)
    preds = module(binary_probs_n2)

    # uses second column as positive class probability
    assert torch.equal(preds, torch.tensor([1, 0, 1, 0, 1], dtype=torch.long))


def test_binary_threshold_rejects_invalid_shape(multiclass_probs: torch.Tensor):
    module = BinaryClassificationThreshold(threshold=0.5)

    with pytest.raises(ValueError, match="expects binary probabilities"):
        module(multiclass_probs)


def test_argmax_decision_returns_class_indices(multiclass_probs: torch.Tensor):
    module = ArgmaxDecision()
    preds = module(multiclass_probs)

    assert isinstance(preds, torch.Tensor)
    assert preds.ndim == 1
    assert preds.shape[0] == multiclass_probs.shape[0]
    assert preds.dtype == torch.long
    assert torch.equal(preds, torch.tensor([1, 0, 2, 2], dtype=torch.long))


@pytest.mark.parametrize("fixture_name", ["binary_probs_n", "binary_probs_n1"])
def test_argmax_decision_rejects_non_multiclass_shapes(
    request: pytest.FixtureRequest,
    fixture_name: str,
):
    probs = request.getfixturevalue(fixture_name)
    module = ArgmaxDecision()

    with pytest.raises(ValueError, match="expects multiclass probabilities"):
        module(probs)


def test_sample_topk_temperature_rejects_invalid_init():
    with pytest.raises(ValueError, match="k must be > 0"):
        SampleTopKTemperature(k=0, temperature=1.0)

    with pytest.raises(ValueError, match="temperature must be > 0"):
        SampleTopKTemperature(k=2, temperature=0.0)


def test_sample_topk_temperature_returns_valid_class_indices(multiclass_probs: torch.Tensor):
    torch.manual_seed(0)
    module = SampleTopKTemperature(k=2, temperature=1.0)

    preds = module(multiclass_probs)

    assert isinstance(preds, torch.Tensor)
    assert preds.ndim == 1
    assert preds.shape[0] == multiclass_probs.shape[0]
    assert preds.dtype == torch.long
    assert torch.all((preds >= 0) & (preds < multiclass_probs.shape[1]))


def test_sample_topk_temperature_with_k_one_matches_argmax(multiclass_probs: torch.Tensor):
    torch.manual_seed(0)
    module = SampleTopKTemperature(k=1, temperature=1.0)

    preds = module(multiclass_probs)
    expected = torch.argmax(multiclass_probs, dim=1)

    assert torch.equal(preds, expected)


@pytest.mark.parametrize("fixture_name", ["binary_probs_n", "binary_probs_n1"])
def test_sample_topk_temperature_rejects_non_multiclass_shapes(
    request: pytest.FixtureRequest,
    fixture_name: str,
):
    probs = request.getfixturevalue(fixture_name)
    module = SampleTopKTemperature(k=2, temperature=1.0)

    with pytest.raises(ValueError, match="expects multiclass probabilities"):
        module(probs)


def test_factory_builds_decision_module():
    spec = DecisionModuleSpec(
        cls=BinaryClassificationThreshold,
        kwargs={"threshold": 0.7},
    )

    module = DecisionModuleFactory.build(spec)

    assert isinstance(module, BinaryClassificationThreshold)
    assert module.threshold == 0.7


def test_factory_rejects_missing_cls():
    spec = DecisionModuleSpec(cls=None)

    with pytest.raises(ValueError, match="must be specified"):
        DecisionModuleFactory.build(spec)


def test_factory_rejects_non_decision_module_cls():
    class NotADecisionModule:
        pass

    spec = DecisionModuleSpec(cls=NotADecisionModule)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a subclass of DecisionModule"):
        DecisionModuleFactory.build(spec)


def test_factory_rejects_both_state_dict_and_path(tmp_path):
    spec = DecisionModuleSpec(cls=BinaryClassificationThreshold, kwargs={"threshold": 0.5})
    path = tmp_path / "dummy.pt"
    torch.save({}, path)

    with pytest.raises(ValueError, match="Only one of state_dict_path or state_dict"):
        DecisionModuleFactory.build(
            spec,
            state_dict_path=str(path),
            state_dict={},
        )


def test_factory_can_load_state_dict():
    original = BinaryClassificationThreshold(threshold=0.8)
    state_dict = original.state_dict()

    spec = DecisionModuleSpec(cls=BinaryClassificationThreshold, kwargs={"threshold": 0.5})
    loaded = DecisionModuleFactory.build(spec, state_dict=state_dict)

    assert isinstance(loaded, BinaryClassificationThreshold)
    # threshold is a plain attribute, not part of state_dict, so constructor value remains
    assert loaded.threshold == 0.5

def test_factory_builds_sample_topk_temperature():
    spec = DecisionModuleSpec(
        cls=SampleTopKTemperature,
        kwargs={"k": 2, "temperature": 0.8},
    )

    module = DecisionModuleFactory.build(spec)

    assert isinstance(module, SampleTopKTemperature)
    assert module.k == 2
    assert module.temperature == 0.8
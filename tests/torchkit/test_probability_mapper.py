from __future__ import annotations

import pytest
import torch

from torchkit.models.probability_mapping._probability_mapper import ProbabilityMapper
from torchkit.models.probability_mapping.classification import ClassificationProbabilityMapper
from torchkit.models.probability_mapping.factory import (
    ProbabilityMapperFactory,
    ProbabilityMapperSpec,
)


class DummyProbabilityMapper(ProbabilityMapper):
    def forward_impl(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + 1.0


class BadShapeProbabilityMapper(ProbabilityMapper):
    def forward_impl(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.unsqueeze(-1)


class BadTypeProbabilityMapper(ProbabilityMapper):
    def forward_impl(self, logits: torch.Tensor):
        return {"not": "a tensor"}


@pytest.fixture
def binary_logits_n() -> torch.Tensor:
    return torch.tensor([2.0, -1.0, 0.5, -0.3, 1.2], dtype=torch.float32)


@pytest.fixture
def binary_logits_n1() -> torch.Tensor:
    return torch.tensor([[2.0], [-1.0], [0.5], [-0.3], [1.2]], dtype=torch.float32)


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


def test_base_probability_mapper_checks_input_type():
    mapper = DummyProbabilityMapper()

    with pytest.raises(ValueError, match="expects `logits` to be a Tensor"):
        mapper([1, 2, 3])


def test_base_probability_mapper_checks_output_type(binary_logits_n1: torch.Tensor):
    mapper = BadTypeProbabilityMapper()

    with pytest.raises(ValueError, match="output of `forward_impl` to be a Tensor"):
        mapper(binary_logits_n1)


def test_base_probability_mapper_checks_output_shape(binary_logits_n1: torch.Tensor):
    mapper = BadShapeProbabilityMapper()

    with pytest.raises(ValueError, match="same shape as input logits"):
        mapper(binary_logits_n1)


def test_base_probability_mapper_passes_through_valid_output(binary_logits_n1: torch.Tensor):
    mapper = DummyProbabilityMapper()
    out = mapper(binary_logits_n1)

    assert out.shape == binary_logits_n1.shape
    assert torch.allclose(out, binary_logits_n1 + 1.0)


@pytest.mark.parametrize("fixture_name", ["binary_logits_n", "binary_logits_n1"])
def test_classification_probability_mapper_binary_shapes(
    request: pytest.FixtureRequest,
    fixture_name: str,
):
    logits = request.getfixturevalue(fixture_name)
    mapper = ClassificationProbabilityMapper()

    probs = mapper(logits)

    assert probs.shape == logits.shape
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    assert torch.allclose(probs, torch.sigmoid(logits))


def test_classification_probability_mapper_multiclass_shape(multiclass_logits: torch.Tensor):
    mapper = ClassificationProbabilityMapper()

    probs = mapper(multiclass_logits)

    assert probs.shape == multiclass_logits.shape
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    assert torch.allclose(probs.sum(dim=1), torch.ones(multiclass_logits.shape[0]))


def test_classification_probability_mapper_binary_two_logit_case():
    logits = torch.tensor(
        [
            [-1.0, 1.0],
            [1.5, -0.5],
            [-0.2, 0.2],
        ],
        dtype=torch.float32,
    )
    mapper = ClassificationProbabilityMapper()

    probs = mapper(logits)

    assert probs.shape == logits.shape
    assert torch.all((probs >= 0.0) & (probs <= 1.0))
    assert torch.allclose(probs.sum(dim=1), torch.ones(logits.shape[0]))


def test_classification_probability_mapper_rejects_invalid_shape():
    logits = torch.randn(2, 3, 4)
    mapper = ClassificationProbabilityMapper()

    with pytest.raises(ValueError, match="expects binary or multiclass logits"):
        mapper(logits)


def test_factory_builds_probability_mapper():
    spec = ProbabilityMapperSpec(
        cls=ClassificationProbabilityMapper,
        kwargs={},
    )

    mapper = ProbabilityMapperFactory.build(spec)

    assert isinstance(mapper, ClassificationProbabilityMapper)


def test_factory_rejects_missing_cls():
    spec = ProbabilityMapperSpec(cls=None)

    with pytest.raises(ValueError, match="must be specified"):
        ProbabilityMapperFactory.build(spec)


def test_factory_rejects_non_mapper_cls():
    class NotAMapper:
        pass

    spec = ProbabilityMapperSpec(cls=NotAMapper)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a subclass of ProbabilityMapper"):
        ProbabilityMapperFactory.build(spec)


def test_factory_rejects_both_state_dict_and_path(tmp_path):
    spec = ProbabilityMapperSpec(cls=ClassificationProbabilityMapper)
    path = tmp_path / "dummy.pt"
    torch.save({}, path)

    with pytest.raises(ValueError, match="Only one of state_dict_path or state_dict"):
        ProbabilityMapperFactory.build(
            spec,
            state_dict_path=str(path),
            state_dict={},
        )


def test_factory_can_load_state_dict():
    original = ClassificationProbabilityMapper()
    state_dict = original.state_dict()

    spec = ProbabilityMapperSpec(cls=ClassificationProbabilityMapper)
    loaded = ProbabilityMapperFactory.build(spec, state_dict=state_dict)

    assert isinstance(loaded, ClassificationProbabilityMapper)
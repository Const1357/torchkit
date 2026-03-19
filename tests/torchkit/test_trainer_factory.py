from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn

from torchkit.train.factory import TrainerFactory, TrainerSpec
from torchkit.train.trainer import Trainer, TrainerConfig

from torchkit.models.Model._model import TorchkitModel
from torchkit.models.Model.factory import TorchkitModelSpec
from torchkit.models.backbone._backbone import Backbone
from torchkit.models.backbone.factory import BackboneSpec
from torchkit.models.head._task_head import TaskHead
from torchkit.models.head.factory import TaskHeadSpec
from torchkit.models.adapters._feature_adapter import FeatureAdapter
from torchkit.models.adapters.factory import FeatureAdapterSpec
from torchkit.models.head_module.factory import HeadModuleSpec

from torchkit.objectives.relational import CELoss
from torchkit.evaluate.select import SelectorEvaluator
from torchkit.evaluate.select.bundle import BundleSelectorEvaluator


# ============================================================
# Dummy model components
# ============================================================

class DummyBackbone(Backbone):
    def __init__(self):
        super().__init__(supported_features=["feat"])

    def _forward_impl(
        self,
        input: dict[str, Tensor],
        *,
        requested_features=None,
        **kwargs,
    ) -> dict[str, Tensor]:
        x = input["x"]
        out = {}
        if "feat" in requested_features:
            out["feat"] = x
        return out


class IdentityAdapter(FeatureAdapter):
    def forward(self, features: Tensor, **kwargs) -> Tensor:
        return features


class LinearLogitsHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        return {"logits": self.linear(x)}


def make_model() -> TorchkitModel:
    backbone = DummyBackbone()
    head = TaskHead(
        required_features="feat",
        feature_adapter=IdentityAdapter(),
        head_module=LinearLogitsHead(in_features=3, out_features=2),
    )
    return TorchkitModel(backbone=backbone, heads={"clf": head})


class DummyEvaluator(SelectorEvaluator):
    def __init__(self):
        super().__init__(name="dummy_eval", direction="maximize", weight=1.0)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("clf/logits",)

    def primary_metric(self, *, inputs: dict[str, object]) -> Tensor:
        logits = self.resolve(inputs, "clf/logits")
        return logits.mean()


class CustomTrainer(Trainer):
    pass


# ============================================================
# Tests
# ============================================================

def test_trainer_factory_build_sane():
    model = make_model()
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    evaluator = DummyEvaluator()

    spec = TrainerSpec(
        cls=Trainer,
        objective=objective,
        selector_evaluator=BundleSelectorEvaluator(
            dataset_evaluator=evaluator,
            batch_evaluator=evaluator,
        ),
        config=TrainerConfig(
            device="cpu",
            max_epochs=3,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
        ),
    )

    trainer = TrainerFactory.build(spec, model=model)

    assert isinstance(trainer, Trainer)
    assert trainer.model is model
    assert trainer.objective is objective
    assert trainer.selector_evaluator is not None
    assert trainer.selector_evaluator.dataset_evaluator is evaluator
    assert trainer.selector_evaluator.batch_evaluator is evaluator
    assert trainer.config.max_epochs == 3
    assert trainer.config.optimizer_kwargs["lr"] == 0.1


def test_trainer_factory_build_rejects_missing_cls():
    model = make_model()
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    spec = TrainerSpec(
        cls=None,
        objective=objective,
    )

    with pytest.raises(ValueError, match="TrainerSpec.cls must be specified"):
        TrainerFactory.build(spec, model=model)


def test_trainer_factory_build_rejects_non_trainer_cls():
    model = make_model()
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    spec = TrainerSpec(
        cls=nn.Linear,  # type: ignore[arg-type]
        objective=objective,
    )

    with pytest.raises(TypeError, match="must be a subclass of Trainer"):
        TrainerFactory.build(spec, model=model)


def test_trainer_factory_build_rejects_missing_objective():
    model = make_model()
    spec = TrainerSpec(
        cls=Trainer,
        objective=None,
    )

    with pytest.raises(ValueError, match="TrainerSpec.objective must be specified"):
        TrainerFactory.build(spec, model=model)


def test_trainer_factory_build_uses_custom_trainer_cls():
    model = make_model()
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    spec = TrainerSpec(
        cls=CustomTrainer,
        objective=objective,
        config=TrainerConfig(device="cpu"),
    )

    trainer = TrainerFactory.build(spec, model=model)

    assert isinstance(trainer, CustomTrainer)


def test_trainer_factory_build_deepcopies_config():
    model = make_model()
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    config = TrainerConfig(
        device="cpu",
        max_epochs=5,
        optimizer_kwargs={"lr": 0.1},
    )
    spec = TrainerSpec(
        cls=Trainer,
        objective=objective,
        config=config,
    )

    trainer = TrainerFactory.build(spec, model=model)

    trainer.config.max_epochs = 99
    trainer.config.optimizer_kwargs["lr"] = 0.9

    assert spec.config.max_epochs == 5
    assert spec.config.optimizer_kwargs["lr"] == 0.1


def test_trainer_factory_build_surfaces_logically_bad_optimizer_kwargs():
    model = make_model()
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    spec = TrainerSpec(
        cls=Trainer,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": "bad"},
        ),
    )

    with pytest.raises(Exception):
        TrainerFactory.build(spec, model=model)


def test_trainer_factory_build_surfaces_logically_bad_scheduler_kwargs():
    model = make_model()
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    spec = TrainerSpec(
        cls=Trainer,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={},  # missing required step_size
        ),
    )

    with pytest.raises(TypeError):
        TrainerFactory.build(spec, model=model)


def test_trainer_factory_build_multiple_times_creates_independent_trainers():
    model1 = make_model()
    model2 = make_model()
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    spec = TrainerSpec(
        cls=Trainer,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
        ),
    )

    trainer1 = TrainerFactory.build(spec, model=model1)
    trainer2 = TrainerFactory.build(spec, model=model2)

    assert trainer1 is not trainer2
    assert trainer1.optimizer is not trainer2.optimizer
    assert trainer1.config is not trainer2.config


def test_trainer_factory_build_from_model_spec_sane():
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    trainer_spec = TrainerSpec(
        cls=Trainer,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            max_epochs=2,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
        ),
    )

    model_spec = TorchkitModelSpec(
        backbone=BackboneSpec(
            cls=DummyBackbone,
            kwargs={},
        ),
        heads={
            "clf": TaskHeadSpec(
                required_features="feat",
                feature_adapter=FeatureAdapterSpec(cls=IdentityAdapter, kwargs={}),
                head_module=HeadModuleSpec(
                    cls=LinearLogitsHead,
                    kwargs={"in_features": 3, "out_features": 2},
                ),
            )
        },
    )

    trainer = TrainerFactory.build_from_model_spec(
        trainer_spec,
        model_spec=model_spec,
    )

    assert isinstance(trainer, Trainer)
    assert isinstance(trainer.model, TorchkitModel)
    assert trainer.config.max_epochs == 2


def test_trainer_factory_build_from_model_spec_deepcopies_inputs():
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    trainer_spec = TrainerSpec(
        cls=Trainer,
        objective=objective,
        config=TrainerConfig(device="cpu"),
    )

    model_spec = TorchkitModelSpec(
        backbone=BackboneSpec(cls=DummyBackbone, kwargs={}),
        heads={
            "clf": TaskHeadSpec(
                required_features="feat",
                feature_adapter=FeatureAdapterSpec(cls=IdentityAdapter, kwargs={}),
                head_module=HeadModuleSpec(
                    cls=LinearLogitsHead,
                    kwargs={"in_features": 3, "out_features": 2},
                ),
            )
        },
    )

    trainer = TrainerFactory.build_from_model_spec(
        trainer_spec,
        model_spec=model_spec,
    )

    # mutate trainer-side config/model independently
    trainer.config.max_epochs = 123

    assert trainer_spec.config.max_epochs != 123
    assert model_spec.heads["clf"].required_features == "feat"

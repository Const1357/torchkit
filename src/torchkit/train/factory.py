from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import copy

from torchkit.models.Model._model import TorchkitModel
from torchkit.models.Model.factory import TorchkitModelFactory, TorchkitModelSpec
from torchkit.objectives import Objective
from torchkit.distributed import DDPStrategy
from torchkit.evaluate.select.bundle import BundleSelectorEvaluator
from torchkit.train.trainer import Trainer, TrainerConfig


@dataclass
class TrainerSpec:
    cls: type[Trainer] | None = Trainer
    objective: Optional[Objective] = None
    selector_evaluator: Optional[BundleSelectorEvaluator] = None
    logging: bool = False
    distributed_strategy: Optional[DDPStrategy] = None
    config: TrainerConfig = field(default_factory=TrainerConfig)


class TrainerFactory:

    @staticmethod
    def build(
        spec: TrainerSpec,
        *,
        model: TorchkitModel,
    ) -> Trainer:
        if spec.cls is None:
            raise ValueError("TrainerSpec.cls must be specified.")

        if not issubclass(spec.cls, Trainer):
            raise TypeError(
                f"TrainerSpec.cls must be a subclass of Trainer, got {spec.cls}."
            )

        if spec.objective is None:
            raise ValueError("TrainerSpec.objective must be specified.")

        trainer = spec.cls(
            model=model,
            objective=spec.objective,
            selector_evaluator=spec.selector_evaluator,
            config=copy.deepcopy(spec.config),
            logging=spec.logging,
            distributed_strategy=copy.deepcopy(spec.distributed_strategy),
        )

        return trainer

    @staticmethod
    def build_from_model_spec(
        spec: TrainerSpec,
        *,
        model_spec: TorchkitModelSpec,
    ) -> Trainer:
        device = spec.config.device if spec.config.device is not None else "cpu"
        model = TorchkitModelFactory.build(
            copy.deepcopy(model_spec),
            device=device,
        )
        return TrainerFactory.build(spec, model=model)

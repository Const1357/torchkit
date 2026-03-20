from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from torchkit.data._dataset import TorchkitDataset
from torchkit.evaluate.select import AccuracySelectorEvaluator
from torchkit.evaluate.select.bundle import BundleSelectorEvaluator
from torchkit.models.Model._model import TorchkitModel
from torchkit.models.Model.factory import TorchkitModelFactory
from torchkit.models.adapters import IdentityAdapter
from torchkit.models.backbone import MLPBackbone
from torchkit.models.decision.classification import ArgmaxDecision
from torchkit.models.head._task_head import TaskHead
from torchkit.models.head_module.classification import ClassifierHeadLinear
from torchkit.models.prediction._prediction_head import PredictionHead
from torchkit.models.probability_mapping.classification import (
    ClassificationProbabilityMapper,
)
from torchkit.objectives.relational import CELoss
from torchkit.train.factory import TrainerSpec
from torchkit.train.trainer import Trainer, TrainerConfig


class XORDataset(TorchkitDataset):
    def __init__(self, repeats: int = 32):
        self._samples: list[tuple[torch.Tensor, int]] = []
        base = [
            (torch.tensor([0.0, 0.0], dtype=torch.float32), 0),
            (torch.tensor([0.0, 1.0], dtype=torch.float32), 1),
            (torch.tensor([1.0, 0.0], dtype=torch.float32), 1),
            (torch.tensor([1.0, 1.0], dtype=torch.float32), 0),
        ]

        for _ in range(repeats):
            for x, y in base:
                self._samples.append((x.clone(), int(y)))

    def __len__(self) -> int:
        return len(self._samples)

    def my_getitem(self, index: int):
        x, y = self._samples[index]
        return {
            "x": x.clone(),
            "y": torch.tensor(y, dtype=torch.long),
        }


def make_xor_model(
    *,
    hidden_dim: int = 16,
) -> TorchkitModel:
    torch.manual_seed(0)

    return TorchkitModel(
        backbone=MLPBackbone(
            input_dim=2,
            hidden_dims=[hidden_dim, hidden_dim],
            output_dim=hidden_dim,
            activation=nn.Tanh(),
            dropout=0.0,
        ),
        heads={
            "clf": TaskHead(
                required_features="features",
                feature_adapter=IdentityAdapter(),
                head_module=ClassifierHeadLinear(
                    input_dim=hidden_dim,
                    num_classes=2,
                ),
                active=True,
            )
        },
        prediction_heads={
            "clf": PredictionHead(
                probability_mapper=ClassificationProbabilityMapper(),
                decision_module=ArgmaxDecision(),
                active=True,
            )
        },
    )


def make_xor_trainer_spec(
    *,
    lr: float,
    max_epochs: int,
    random_seed: int = 0,
) -> TrainerSpec:
    return TrainerSpec(
        cls=Trainer,
        objective=CELoss(
            input_path="clf/logits",
            target_path="batch/y",
            reduction="mean",
            name="xor_ce",
        ),
        selector_evaluator=BundleSelectorEvaluator(
            dataset_evaluator=AccuracySelectorEvaluator(
                score_key="clf/logits",
                target_key="batch/y",
                predictions_key="clf/predictions",
                name="xor_accuracy",
            )
        ),
        config=TrainerConfig(
            device="cpu",
            random_seed=random_seed,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": lr},
            max_epochs=max_epochs,
            early_stopping_patience=None,
            keep_history_on_reset=False,
        ),
    )


def make_xor_loader(
    dataset: TorchkitDataset | Subset,
    *,
    shuffle: bool,
    batch_size: int = 16,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def xor_accuracy(
    model: TorchkitModel,
    dataset: TorchkitDataset | Subset,
    *,
    indices: Optional[Iterable[int]] = None,
) -> float:
    eval_ds = Subset(dataset, list(indices)) if indices is not None else dataset
    loader = DataLoader(eval_ds, batch_size=len(eval_ds), shuffle=False)
    batch = next(iter(loader))
    pred = model.predict(batch, "clf", return_raw_head_outputs=True)
    predictions = pred["clf"]["predictions"]
    targets = batch["y"]
    return float((predictions == targets).float().mean().item())


def rebuild_model_accuracy_from_spec(
    model: TorchkitModel,
    dataset: TorchkitDataset | Subset,
) -> float:
    rebuilt = TorchkitModelFactory.build(
        model.to_spec(),
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        device="cpu",
    )
    return xor_accuracy(rebuilt, dataset)

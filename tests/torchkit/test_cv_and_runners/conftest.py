from __future__ import annotations

from typing import Any, Optional

import torch
import pytest
from torch import Tensor, nn

from torchkit.data._dataset import TorchkitDataset
from torchkit.data.split import StratifiedKFold, GroupKFold, StratifiedGroupKFold

from torchkit.models.backbone._backbone import Backbone
from torchkit.models.backbone.factory import BackboneSpec
from torchkit.models.Model.factory import TorchkitModelSpec
from torchkit.models.head.factory import TaskHeadSpec
from torchkit.models.adapters._feature_adapter import IdentityAdapter
from torchkit.models.adapters.factory import FeatureAdapterSpec
from torchkit.models.head_module.factory import HeadModuleSpec
from torchkit.models.prediction.factory import PredictionHeadSpec
from torchkit.models.calibration._calibrator import Calibrator
from torchkit.models.calibration.factory import CalibratorSpec

from torchkit.train.factory import TrainerSpec
from torchkit.train.trainer import Trainer, TrainerConfig

from torchkit.objectives.relational import CELoss
from torchkit.evaluate._evaluator import Evaluator
from torchkit.evaluate.classification_evaluator import ClassificationEvaluator

from torchkit.train.cv.optuna_search_cv import OptunaSearchCV
from torchkit.train.cv.nested_optuna_search_cv import NestedOptunaSearchCV


class DeterministicBackbone(Backbone):
    """
    Returns one exposed feature map "features" = scale_factor * x.
    Includes a dummy trainable parameter so optimizer/state_dict flow is real.
    """
    def __init__(self, scale_factor: float = 1.0):
        super().__init__(supported_features={"features"})
        self.scale_factor = float(scale_factor)
        self._dummy = nn.Parameter(torch.tensor(0.0))

    def _forward_impl(
        self,
        input: dict[str, Any],
        *,
        requested_features=None,
        **kwargs,
    ) -> dict[str, Tensor]:
        x = input["x"]
        features = self.scale_factor * x + (0.0 * self._dummy)
        return {"features": features}


class DirectBinaryLogitsHead(nn.Module):
    """
    Deterministic binary logits from the first two feature dimensions.
    For the dataset below:
      - scale_factor = +1.0 => correct classification
      - scale_factor = -1.0 => inverted classification
    """
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError(f"Expected x of shape (N, D>=2), got {tuple(x.shape)}.")
        logits = torch.stack([x[:, 0], x[:, 1]], dim=1)
        return {"logits": logits}


class RecordingIdentityCalibrator(Calibrator):
    """
    Identity calibrator that records fit calls in buffers so that calibration
    survives state_dict save/load and can be verified after reconstruction.
    """
    def __init__(self, active: bool = True):
        super().__init__(active=active)
        self.register_buffer("fit_calls", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_num_samples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("last_num_classes", torch.tensor(0, dtype=torch.long))

    def forward_impl(self, logits: Tensor) -> Tensor:
        return logits

    def fit_impl(self, logits: Tensor, targets: Tensor) -> None:
        self.fit_calls += 1
        self.last_num_samples.fill_(int(logits.shape[0]))
        self.last_num_classes.fill_(int(logits.shape[1]) if logits.ndim == 2 else 1)


class TinyClassificationDataset(TorchkitDataset):
    """
    16 samples total, 8 groups, each group has exactly:
      - one class-0 sample: x=[2,0,0]
      - one class-1 sample: x=[0,2,0]

    Supports:
      - StratifiedKFold
      - GroupKFold
      - StratifiedGroupKFold
    """
    def __init__(self):
        self._xs: list[Tensor] = []
        self._ys: list[int] = []

        for _group in range(8):
            self._xs.append(torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32))
            self._ys.append(0)

            self._xs.append(torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32))
            self._ys.append(1)

    def __len__(self) -> int:
        return len(self._xs)

    def my_getitem(self, index) -> dict[str, Any]:
        return {
            "x": self._xs[index].clone(),
            "y": torch.tensor(self._ys[index], dtype=torch.long),
        }


class ErrorRateEvaluator(Evaluator):
    """
    Same prediction surface as ClassificationEvaluator, but primary metric is minimized.
    """
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        name: str = "error_rate",
    ):
        super().__init__(
            name=name,
            primary_metric="error_rate",
            direction="minimize",
            weight=1.0,
        )
        self.score_key = score_key
        self.target_key = target_key

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.score_key, self.target_key)

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        logits = self.resolve(inputs, self.score_key).detach()
        targets = self.resolve(inputs, self.target_key).detach()

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected binary logits of shape (N,2), got {tuple(logits.shape)}.")
        if targets.ndim != 1:
            raise ValueError(f"Expected targets of shape (N,), got {tuple(targets.shape)}.")

        preds = torch.argmax(logits, dim=1)
        error_rate = float((preds != targets).float().mean())
        return {"error_rate": error_rate}


def make_labels_and_groups() -> tuple[list[int], list[int]]:
    y: list[int] = []
    groups: list[int] = []

    for g in range(8):
        y.extend([0, 1])
        groups.extend([g, g])

    return y, groups


def make_model_spec(*, scale_factor: float = 1.0) -> TorchkitModelSpec:
    return TorchkitModelSpec(
        backbone=BackboneSpec(
            cls=DeterministicBackbone,
            kwargs={"scale_factor": scale_factor},
        ),
        heads={
            "clf": TaskHeadSpec(
                required_features="features",
                feature_adapter=FeatureAdapterSpec(cls=IdentityAdapter, kwargs={}),
                head_module=HeadModuleSpec(cls=DirectBinaryLogitsHead, kwargs={}),
                active=True,
            )
        },
        prediction_heads={
            "clf": PredictionHeadSpec(
                calibrator=CalibratorSpec(
                    cls=RecordingIdentityCalibrator,
                    kwargs={},
                    active=True,
                ),
                probability_mapper=None,
                decision_module=None,
                active=True,
            )
        },
    )


def make_trainer_spec(*, evaluator: Evaluator, max_epochs: int = 2) -> TrainerSpec:
    return TrainerSpec(
        cls=Trainer,
        objective=CELoss(
            input_path="clf/logits",
            target_path="batch/y",
            reduction="mean",
        ),
        dataset_evaluator=evaluator,
        batch_evaluator=None,
        config=TrainerConfig(
            device="cpu",
            random_seed=0,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.0},  # deterministic / frozen behavior
            max_epochs=max_epochs,
            early_stopping_patience=None,
            keep_history_on_reset=False,
        ),
    )


def make_optuna_search_cv(
    *,
    model_spec: TorchkitModelSpec,
    trainer_spec: TrainerSpec,
    splitter_cls,
    parameter_grid: dict[str, tuple[list, str]],
    tmp_path,
    n_trials: int = 1,
    max_trial_attempts: int = 5,
    n_splits: int = 2,
    random_state: Optional[int] = None,
    calibrate: bool = True,
    keep_final_model_state_dict_cpu: bool = True,
) -> OptunaSearchCV:
    return OptunaSearchCV(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        parameter_grid=parameter_grid,
        splitter_cls=splitter_cls,
        dataloader_factory=lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=2, shuffle=shuffle),
        n_trials=n_trials,
        max_trial_attempts=max_trial_attempts,
        n_splits=n_splits,
        shuffle=False,
        random_state=random_state,
        calibrate=calibrate,
        final_model_dir=str(tmp_path),
        keep_final_model_state_dict_cpu=keep_final_model_state_dict_cpu,
    )


def make_nested_cv(
    *,
    model_spec: TorchkitModelSpec,
    trainer_spec: TrainerSpec,
    outer_splitter_cls,
    inner_splitter_cls,
    parameter_grid: dict[str, tuple[list, str]],
    tmp_path,
    n_trials: int = 1,
    max_trial_attempts: int = 5,
    k_outer: int = 2,
    k_inner: int = 2,
    random_state: Optional[int] = None,
    calibrate: bool = True,
    keep_final_model_state_dict_cpu: bool = True,
) -> NestedOptunaSearchCV:
    return NestedOptunaSearchCV(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        parameter_grid=parameter_grid,
        outer_splitter_cls=outer_splitter_cls,
        inner_splitter_cls=inner_splitter_cls,
        dataloader_factory=lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=2, shuffle=shuffle),
        n_trials=n_trials,
        max_trial_attempts=max_trial_attempts,
        k_outer=k_outer,
        k_inner=k_inner,
        shuffle_outer=False,
        shuffle_inner=False,
        random_state=random_state,
        calibrate=calibrate,
        final_model_dir=str(tmp_path),
        keep_final_model_state_dict_cpu=keep_final_model_state_dict_cpu,
    )


@pytest.fixture
def tiny_dataset() -> TinyClassificationDataset:
    return TinyClassificationDataset()


@pytest.fixture
def tiny_labels_groups() -> tuple[list[int], list[int]]:
    return make_labels_and_groups()
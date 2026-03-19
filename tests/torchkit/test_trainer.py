from __future__ import annotations

import copy

import optuna
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from torchkit.train.trainer import Trainer, TrainerConfig, _move_to_device
from torchkit.models.Model._model import TorchkitModel
from torchkit.models.backbone._backbone import Backbone
from torchkit.models.head._task_head import TaskHead
from torchkit.models.prediction._prediction_head import PredictionHead

from torchkit.models.adapters._feature_adapter import FeatureAdapter
from torchkit.models.calibration._calibrator import Calibrator
from torchkit.models.probability_mapping._probability_mapper import ProbabilityMapper
from torchkit.models.decision._decision_module import DecisionModule

from torchkit.objectives.relational import CELoss
from torchkit.objectives.Multitask import MultitaskObjective
from torchkit.evaluate.select import SelectorEvaluator
from torchkit.evaluate.select.bundle import BundleSelectorEvaluator


# ============================================================
# Minimal reusable model components
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


class NoOpCalibrator(Calibrator):
    def __init__(self, *, active: bool = True):
        super().__init__(active=active)

    def fit_impl(self, logits: Tensor, targets: Tensor) -> None:
        return None

    def forward_impl(self, logits: Tensor) -> Tensor:
        return logits


class SoftmaxProbabilityMapper(ProbabilityMapper):
    def forward_impl(self, logits: Tensor) -> Tensor:
        return torch.softmax(logits, dim=1)


class ArgmaxDecisionModule(DecisionModule):
    def forward_impl(self, probs: Tensor) -> Tensor:
        return torch.argmax(probs, dim=1)


def make_classification_model(*, with_prediction_head: bool = False) -> TorchkitModel:
    backbone = DummyBackbone()

    head = TaskHead(
        required_features="feat",
        feature_adapter=IdentityAdapter(),
        head_module=LinearLogitsHead(in_features=3, out_features=2),
    )

    if not with_prediction_head:
        return TorchkitModel(backbone=backbone, heads={"clf": head})

    phead = PredictionHead(
        calibrator=NoOpCalibrator(active=True),
        probability_mapper=SoftmaxProbabilityMapper(),
        decision_module=ArgmaxDecisionModule(),
        active=True,
    )

    return TorchkitModel(
        backbone=backbone,
        heads={"clf": head},
        prediction_heads={"clf": phead},
    )


# ============================================================
# Datasets
# ============================================================

class DictClassificationDataset(Dataset):
    def __init__(self, n: int = 12):
        super().__init__()
        xs = []
        ys = []
        for i in range(n):
            if i % 2 == 0:
                xs.append(torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32))
                ys.append(0)
            else:
                xs.append(torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32))
                ys.append(1)
        self.x = xs
        self.y = ys

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "x": self.x[idx],
            "y": torch.tensor(self.y[idx], dtype=torch.long),
        }


class NonDictDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Tensor:
        return torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)


class MissingXDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "y": torch.tensor(idx % 2, dtype=torch.long),
        }


class EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        raise IndexError


# ============================================================
# Evaluators for trainer tests
# ============================================================

class DatasetAccuracyEvaluator(SelectorEvaluator):
    def __init__(self):
        super().__init__(name="dataset_accuracy", direction="maximize", weight=1.0)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("clf/logits", "batch/y")

    def primary_metric(self, *, inputs: dict[str, object]) -> Tensor:
        logits = self.resolve(inputs, "clf/logits")
        targets = self.resolve(inputs, "batch/y")
        preds = torch.argmax(logits, dim=1)
        return (preds == targets).float().mean()


class BatchAccuracyEvaluator(SelectorEvaluator):
    def __init__(self):
        super().__init__(name="batch_accuracy", direction="maximize", weight=1.0)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("clf/logits", "batch/y")

    def primary_metric(self, *, inputs: dict[str, object]) -> Tensor:
        logits = self.resolve(inputs, "clf/logits")
        targets = self.resolve(inputs, "batch/y")
        preds = torch.argmax(logits, dim=1)
        return (preds == targets).float().mean()


class BadBatchEvaluator(SelectorEvaluator):
    def __init__(self):
        super().__init__(name="bad_batch", direction="maximize", weight=1.0)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("clf/logits",)

    def primary_metric(self, *, inputs: dict[str, object]):
        return "not a dict"


class BadDatasetEvaluator(SelectorEvaluator):
    def __init__(self):
        super().__init__(name="bad_dataset", direction="maximize", weight=1.0)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("clf/logits",)

    def primary_metric(self, *, inputs: dict[str, object]):
        return "not a dict"


# ============================================================
# Trial stub
# ============================================================

class DummyTrial:
    def __init__(self, *, prune_on_report: bool = False):
        self.reports: list[tuple[float, int]] = []
        self.prune_on_report = prune_on_report

    def report(self, value: float, step: int) -> None:
        self.reports.append((value, step))

    def should_prune(self) -> bool:
        return self.prune_on_report


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def train_loader() -> DataLoader:
    return DataLoader(DictClassificationDataset(12), batch_size=4, shuffle=False)


@pytest.fixture
def val_loader() -> DataLoader:
    return DataLoader(DictClassificationDataset(8), batch_size=4, shuffle=False)


@pytest.fixture
def trainer() -> Trainer:
    model = make_classification_model(with_prediction_head=True)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    config = TrainerConfig(
        device="cpu",
        max_epochs=3,
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.1},
    )
    return Trainer(
        model=model,
        objective=objective,
        selector_evaluator=BundleSelectorEvaluator(
            dataset_evaluator=DatasetAccuracyEvaluator(),
            batch_evaluator=BatchAccuracyEvaluator(),
        ),
        config=config,
    )


# ============================================================
# Core sane passes
# ============================================================

def test_trainer_fit_end_to_end_with_validation(
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    out = trainer.fit(train_loader, val_loader)

    assert out is trainer
    assert trainer._fit_called_at_least_once is True
    assert len(trainer.state.train_logs) >= 1
    assert len(trainer.state.val_logs) >= 1
    assert trainer.state.best_epoch is not None
    assert trainer.state.best_metric is not None
    assert trainer.state.best_state_dict_cpu is not None
    if hasattr(trainer.model, "active_calibrator_names") and "clf" in trainer.model.active_calibrator_names:
        assert "clf" in trainer.state.oof_logits
        assert "clf" in trainer.state.oof_targets
        assert trainer.state.oof_logits["clf"].shape[0] == len(val_loader.dataset)
        assert trainer.state.oof_targets["clf"].shape[0] == len(val_loader.dataset)
    else:
        assert trainer.state.oof_logits == {}
        assert trainer.state.oof_targets == {}
    assert len(trainer.history) == 1


def test_trainer_fit_end_to_end_without_validation(
    train_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            max_epochs=2,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
        ),
    )

    trainer.fit(train_loader, val_loader=None)

    assert len(trainer.state.train_logs) == 2
    assert len(trainer.state.val_logs) == 0
    assert trainer.state.best_epoch is None
    assert trainer.state.best_metric is None
    assert len(trainer.history) == 1


# ============================================================
# get/set/reset API
# ============================================================

def test_trainer_get_params_returns_config_copy(trainer: Trainer):
    params = trainer.get_params(deep=True)

    assert isinstance(params, dict)
    assert params["max_epochs"] == trainer.config.max_epochs
    assert params["optimizer_kwargs"]["lr"] == trainer.config.optimizer_kwargs["lr"]

    params["optimizer_kwargs"]["lr"] = 999.0
    assert trainer.config.optimizer_kwargs["lr"] != 999.0


def test_trainer_set_params_updates_and_rebuilds(trainer: Trainer):
    old_optimizer = trainer.optimizer

    trainer.set_params(
        max_epochs=9,
        optimizer_kwargs={"lr": 0.05},
    )

    assert trainer.config.max_epochs == 9
    assert trainer.config.optimizer_kwargs["lr"] == 0.05
    assert trainer.optimizer is not old_optimizer


def test_trainer_set_params_unknown_key_raises(trainer: Trainer):
    with pytest.raises(ValueError, match="Unknown TrainerConfig parameter"):
        trainer.set_params(not_a_real_param=123)


def test_trainer_set_params_device_none_raises(trainer: Trainer):
    with pytest.raises(ValueError, match="device cannot be None"):
        trainer.set_params(device=None)


def test_trainer_reset_state_restores_weights_and_clears_state(
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    initial_sd = copy.deepcopy(trainer._get_model_state_dict_cpu())

    trainer.fit(train_loader, val_loader)
    assert len(trainer.state.train_logs) > 0

    trainer.reset_state()

    after_reset_sd = trainer._get_model_state_dict_cpu()
    for k in initial_sd:
        assert torch.allclose(initial_sd[k], after_reset_sd[k])

    assert trainer.state.epoch == 0
    assert trainer.state.best_epoch is None
    assert trainer.state.best_metric is None
    assert trainer.state.train_logs == []
    assert trainer.state.val_logs == []


def test_trainer_reset_state_respects_keep_history_on_reset(
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            max_epochs=1,
            keep_history_on_reset=True,
        ),
    )

    trainer.fit(train_loader, val_loader)
    assert len(trainer.history) == 1

    trainer.reset_state()
    assert len(trainer.history) == 1


def test_trainer_reset_config_restores_base_config(trainer: Trainer):
    trainer.set_params(max_epochs=17, optimizer_kwargs={"lr": 0.07})
    assert trainer.config.max_epochs == 17

    trainer.reset_config()

    assert trainer.config.max_epochs == trainer._base_config.max_epochs
    assert trainer.config.optimizer_kwargs == trainer._base_config.optimizer_kwargs


def test_trainer_detach_model_returns_same_object(trainer: Trainer):
    detached = trainer.detach_model()
    assert detached is trainer.model


# ============================================================
# Validation / logging details
# ============================================================

def test_validate_one_epoch_logs_dataset_and_batch_metrics(
    trainer: Trainer,
    val_loader: DataLoader,
):
    log = trainer._validate_one_epoch(val_loader, epoch=1)

    assert "val_loss" in log
    assert "val/dataset_accuracy" in log
    assert "val_batch/batch_accuracy" in log
    assert "__selection_score__" in log


def test_train_one_epoch_logs_multitask_component_losses(
    train_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)

    ce1 = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
        weight=1.0,
    )
    ce2 = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
        weight=0.5,
        name="aux_ce",
    )
    objective = MultitaskObjective(ce1, ce2, name="multi")

    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            max_epochs=1,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
        ),
    )

    log = trainer._train_one_epoch(train_loader, epoch=1)

    assert "train_loss" in log
    assert any(k.startswith("train_loss/") for k in log.keys())


def test_dataset_evaluator_drives_best_metric_selection(
    trainer: Trainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    trainer.fit(train_loader, val_loader)

    assert trainer.state.best_metric is not None
    last_val_log = trainer.state.val_logs[-1]
    assert "val/accuracy" in last_val_log or "val_loss" in last_val_log


def test_bad_batch_evaluator_raises_in_validation(
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        selector_evaluator=BundleSelectorEvaluator(batch_evaluator=BadBatchEvaluator()),
        config=TrainerConfig(device="cpu", max_epochs=1),
    )

    with pytest.raises(TypeError, match="Tensor"):
        trainer.fit(train_loader, val_loader)


def test_bad_dataset_evaluator_raises_in_validation(
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        selector_evaluator=BundleSelectorEvaluator(dataset_evaluator=BadDatasetEvaluator()),
        config=TrainerConfig(device="cpu", max_epochs=1),
    )

    with pytest.raises(TypeError, match="Tensor"):
        trainer.fit(train_loader, val_loader)


# ============================================================
# Early stopping / scheduler / optuna hooks
# ============================================================

def test_trainer_early_stopping_stops_before_max_epochs(
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        selector_evaluator=BundleSelectorEvaluator(dataset_evaluator=DatasetAccuracyEvaluator()),
        config=TrainerConfig(
            device="cpu",
            max_epochs=10,
            early_stopping_patience=0,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.0},
        ),
    )

    trainer.fit(train_loader, val_loader)

    assert trainer.state.epoch < 10


def test_trainer_scheduler_step_non_metric(
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            max_epochs=2,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 1, "gamma": 0.5},
        ),
    )

    trainer.fit(train_loader, val_loader)
    lrs = [group["lr"] for group in trainer.optimizer.param_groups]
    assert all(lr < 0.1 for lr in lrs)


def test_trainer_scheduler_step_metric_aware(
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            max_epochs=2,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
            scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_kwargs={"patience": 0, "factor": 0.5},
        ),
    )

    trainer.fit(train_loader, val_loader)
    assert trainer.scheduler is not None


def test_trainer_maybe_report_to_trial_records_reports():
    trial = DummyTrial(prune_on_report=False)
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(model=model, objective=objective, config=TrainerConfig(device="cpu"))

    trainer.maybe_report_to_trial(trial, value=1.23, step=4)

    assert trial.reports == [(1.23, 4)]


def test_trainer_maybe_report_to_trial_can_prune():
    trial = DummyTrial(prune_on_report=True)
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(model=model, objective=objective, config=TrainerConfig(device="cpu"))

    with pytest.raises(optuna.TrialPruned):
        trainer.maybe_report_to_trial(trial, value=1.23, step=4)


# ============================================================
# Broken-but-valid configs / loaders
# ============================================================

def test_trainer_init_with_bad_optimizer_kwargs_raises():
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    with pytest.raises(Exception):
        Trainer(
            model=model,
            objective=objective,
            config=TrainerConfig(
                device="cpu",
                optimizer_cls=torch.optim.SGD,
                optimizer_kwargs={"lr": "not-a-float"},
            ),
        )


def test_trainer_init_with_bad_scheduler_kwargs_raises():
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    with pytest.raises(TypeError):
        Trainer(
            model=model,
            objective=objective,
            config=TrainerConfig(
                device="cpu",
                optimizer_cls=torch.optim.SGD,
                optimizer_kwargs={"lr": 0.1},
                scheduler_cls=torch.optim.lr_scheduler.StepLR,
                scheduler_kwargs={},  # missing step_size
            ),
        )


def test_trainer_fit_rejects_nonpositive_max_epochs(
    trainer: Trainer,
    train_loader: DataLoader,
):
    with pytest.raises(ValueError, match="max_epochs must be > 0"):
        trainer.fit(train_loader, val_loader=None, max_epochs=0)


def test_trainer_fit_rejects_negative_patience(
    trainer: Trainer,
    train_loader: DataLoader,
):
    with pytest.raises(ValueError, match="early_stopping_patience must be >=0"):
        trainer.fit(train_loader, val_loader=None, early_stopping_patience=-1)


def test_train_one_epoch_rejects_nondict_batches():
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(model=model, objective=objective, config=TrainerConfig(device="cpu"))

    loader = DataLoader(NonDictDataset(), batch_size=2)

    with pytest.raises(TypeError, match="Expected batch as dict"):
        trainer._train_one_epoch(loader, epoch=1)


def test_train_one_epoch_rejects_missing_x():
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(model=model, objective=objective, config=TrainerConfig(device="cpu"))

    loader = DataLoader(MissingXDataset(), batch_size=2)

    with pytest.raises(KeyError, match="contain the 'x' key"):
        trainer._train_one_epoch(loader, epoch=1)


def test_train_one_epoch_rejects_empty_loader():
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(model=model, objective=objective, config=TrainerConfig(device="cpu"))

    loader = DataLoader(EmptyDataset(), batch_size=2)

    with pytest.raises(ValueError, match="train_loader produced 0 batches"):
        trainer._train_one_epoch(loader, epoch=1)


def test_validate_one_epoch_rejects_empty_loader():
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(model=model, objective=objective, config=TrainerConfig(device="cpu"))

    loader = DataLoader(EmptyDataset(), batch_size=2)

    with pytest.raises(ValueError, match="val_loader produced 0 batches"):
        trainer._validate_one_epoch(loader, epoch=1)


def test_trainer_can_load_initial_state_from_path(train_loader: DataLoader, tmp_path):
    model = make_classification_model(with_prediction_head=False)
    init_path = tmp_path / "init_state.pt"
    torch.save(model.state_dict(), init_path)

    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )

    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            max_epochs=1,
            model_initial_state_path=str(init_path),
        ),
    )

    trainer.fit(train_loader, val_loader=None)
    trainer.reset_state()

    reloaded_sd = trainer._get_model_state_dict_cpu()
    saved_sd = torch.load(init_path, map_location="cpu")
    for k in saved_sd:
        assert torch.allclose(saved_sd[k], reloaded_sd[k])

class NestedBatchDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "x": torch.randn(3),
            "nested": {
                "a": torch.randn(2),
                "b": [torch.randn(1), (torch.randn(1), "text")]
            },
            "y": torch.tensor(idx % 2)
        }


def test_move_to_device_nested_structures():
    model = nn.Linear(3, 2)
    objective = CELoss(input_path="clf/logits", target_path="batch/y")

    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(device="cpu")
    )

    batch = NestedBatchDataset()[0]
    moved = _move_to_device(batch, device="cpu")

    assert isinstance(moved["nested"]["b"][0], torch.Tensor)
    assert moved["nested"]["b"][1][1] == "text"


# -------------------------------------------------------------------
# set_params branches
# -------------------------------------------------------------------

def test_set_params_rebuilds_scaler():
    model = nn.Linear(3, 2)
    objective = CELoss(input_path="clf/logits", target_path="batch/y")

    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(device="cpu", use_amp=False)
    )

    old_scaler = trainer._scaler

    trainer.set_params(use_amp=True)

    assert trainer._scaler is not old_scaler


def test_set_params_remove_scheduler():
    model = nn.Linear(3, 2)
    objective = CELoss(input_path="clf/logits", target_path="batch/y")

    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(device="cpu")
    )

    trainer.set_params(scheduler_cls=None)

    assert trainer.scheduler is None


# -------------------------------------------------------------------
# reset_state branches
# -------------------------------------------------------------------

def test_reset_state_config_and_history():
    model = nn.Linear(3, 2)
    objective = CELoss(input_path="clf/logits", target_path="batch/y")

    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(device="cpu")
    )

    trainer.history.append({"train_loss": 1.0})

    trainer.reset_state(reset_config=True, clear_history=False)

    assert len(trainer.history) == 1


# -------------------------------------------------------------------
# Validation input guards
# -------------------------------------------------------------------

class NonTensorXDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"x": "not_tensor", "y": torch.tensor(1)}


class ScalarXDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {"x": torch.tensor(1.0), "y": torch.tensor(1)}


def make_loader(dataset):
    return DataLoader(dataset, batch_size=2)


def test_validate_rejects_non_tensor_x():
    model = nn.Linear(3, 2)
    objective = CELoss(input_path="clf/logits", target_path="batch/y")

    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(device="cpu")
    )

    loader = make_loader(NonTensorXDataset())

    try:
        trainer._validate_one_epoch(loader, epoch=1)
    except TypeError:
        pass
    else:
        assert False


# -------------------------------------------------------------------
# Calibration target discovery branches
# -------------------------------------------------------------------

class NestedTargetDataset(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return {
            "x": torch.randn(3),
            "clf": {"y": torch.tensor(idx % 2)}
        }


class NonTensorXDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "x": f"sample-{idx}",
            "y": torch.tensor(idx % 2, dtype=torch.long),
        }


class ScalarXDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "x": torch.tensor(float(idx), dtype=torch.float32),
            "y": torch.tensor(idx % 2, dtype=torch.long),
        }


class NestedCalibrationTargetDataset(Dataset):
    def __len__(self) -> int:
        return 6

    def __getitem__(self, idx: int) -> dict[str, object]:
        y = torch.tensor(idx % 2, dtype=torch.long)
        x = (
            torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
            if idx % 2 == 0
            else torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
        )
        return {
            "x": x,
            "target": y,
            "clf": {"y": y},
        }


class FlatCalibrationTargetDataset(Dataset):
    def __len__(self) -> int:
        return 6

    def __getitem__(self, idx: int) -> dict[str, object]:
        y = torch.tensor(idx % 2, dtype=torch.long)
        x = (
            torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32)
            if idx % 2 == 0
            else torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32)
        )
        return {
            "x": x,
            "target": y,
            "clf/y": y,
        }


def test_move_to_device_recurses_into_nested_structures():
    nested = {
        "a": torch.tensor([1.0]),
        "b": [torch.tensor([2.0]), (torch.tensor([3.0]), "x")],
        "c": 7,
    }

    moved = _move_to_device(nested, torch.device("cpu"))

    assert moved["a"].device.type == "cpu"
    assert moved["b"][0].device.type == "cpu"
    assert moved["b"][1][0].device.type == "cpu"
    assert moved["b"][1][1] == "x"
    assert moved["c"] == 7


def test_trainer_set_params_use_amp_rebuilds_scaler(trainer: Trainer):
    old_scaler = trainer._scaler

    trainer.set_params(use_amp=True)

    assert trainer.config.use_amp is True
    assert trainer._scaler is not old_scaler
    assert trainer._scaler.is_enabled() is False


def test_trainer_set_params_scheduler_cls_none_removes_scheduler(
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            max_epochs=1,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
            scheduler_cls=torch.optim.lr_scheduler.StepLR,
            scheduler_kwargs={"step_size": 1, "gamma": 0.5},
        ),
    )

    trainer.fit(train_loader, val_loader)
    assert trainer.scheduler is not None

    trainer.set_params(scheduler_cls=None)

    assert trainer.scheduler is None


def test_trainer_reset_state_with_reset_config_preserves_history_when_requested(
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(
            device="cpu",
            max_epochs=1,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
        ),
    )

    trainer.set_params(max_epochs=7, optimizer_kwargs={"lr": 0.02})
    trainer.fit(train_loader, val_loader)
    assert len(trainer.history) == 1

    trainer.reset_state(reset_config=True, clear_history=False)

    assert trainer.config.max_epochs == trainer._base_config.max_epochs
    assert trainer.config.optimizer_kwargs == trainer._base_config.optimizer_kwargs
    assert len(trainer.history) == 1
    assert trainer.state.train_logs == []
    assert trainer.state.val_logs == []


def test_validate_one_epoch_rejects_non_tensor_x():
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(device="cpu"),
    )

    loader = DataLoader(NonTensorXDataset(), batch_size=None)

    with pytest.raises(TypeError, match=r"'x' is supposed to be a Tensor"):
        trainer._validate_one_epoch(loader, epoch=1)


def test_validate_one_epoch_rejects_scalar_x():
    model = make_classification_model(with_prediction_head=False)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/y",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(device="cpu"),
    )

    loader = DataLoader(ScalarXDataset(), batch_size=None)

    with pytest.raises(ValueError, match=r"batch\['x'\] is scalar"):
        trainer._validate_one_epoch(loader, epoch=1)


def test_validate_oof_targets_found_in_nested_task_dict():
    model = make_classification_model(with_prediction_head=True)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/target",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(device="cpu", max_epochs=1),
    )

    loader = DataLoader(NestedCalibrationTargetDataset(), batch_size=2, shuffle=False)
    log = trainer._validate_one_epoch(loader, epoch=1)

    assert "val_loss" in log
    assert "clf" in trainer.state.oof_targets
    assert trainer.state.oof_targets["clf"].shape[0] == len(loader.dataset)


def test_validate_oof_targets_found_in_flat_task_key():
    model = make_classification_model(with_prediction_head=True)
    objective = CELoss(
        input_path="clf/logits",
        target_path="batch/target",
        reduction="mean",
    )
    trainer = Trainer(
        model=model,
        objective=objective,
        config=TrainerConfig(device="cpu", max_epochs=1),
    )

    loader = DataLoader(FlatCalibrationTargetDataset(), batch_size=2, shuffle=False)
    log = trainer._validate_one_epoch(loader, epoch=1)

    assert "val_loss" in log
    assert "clf" in trainer.state.oof_targets
    assert trainer.state.oof_targets["clf"].shape[0] == len(loader.dataset)

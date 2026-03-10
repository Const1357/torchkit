from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Optional

import optuna
import pytest
import torch
from torch import Tensor, nn

from torchkit.data._dataset import TorchkitDataset
from torchkit.data.split import StratifiedKFold, GroupKFold, StratifiedGroupKFold

from torchkit.models.backbone._backbone import Backbone
from torchkit.models.backbone.factory import BackboneSpec
from torchkit.models.Model._model import TorchkitModel
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
from torchkit.train.nested_optuna_search_cv import NestedOptunaSearchCV, NestedCVResult

from torchkit.objectives.relational import CELoss
from torchkit.evaluate._evaluator import Evaluator
from torchkit.evaluate.classification_evaluator import ClassificationEvaluator


# ============================================================
# Minimal real components for deterministic CV behavior
# ============================================================

class DeterministicBackbone(Backbone):
    """
    Returns a single feature map "features" equal to scale_factor * x.
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
      - scale_factor = +1.0 => correct
      - scale_factor = -1.0 => inverted
    """
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError(f"Expected x of shape (N, D>=2), got {tuple(x.shape)}.")
        logits = torch.stack([x[:, 0], x[:, 1]], dim=1)
        return {"logits": logits}


class RecordingIdentityCalibrator(Calibrator):
    """
    Identity calibrator that records fit calls in state_dict-compatible buffers.
    This lets us verify that calibration happened and on how many OOF samples.
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
        if logits.ndim == 2:
            self.last_num_classes.fill_(int(logits.shape[1]))
        else:
            self.last_num_classes.fill_(1)


class TinyClassificationDataset(TorchkitDataset):
    """
    16 samples total, 8 groups, each group has exactly:
      - one class-0 sample: x=[2,0,0]
      - one class-1 sample: x=[0,2,0]

    This supports:
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


# ============================================================
# Helpers
# ============================================================

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
            optimizer_kwargs={"lr": 0.0},  # freeze behavior for deterministic CV
            max_epochs=max_epochs,
            early_stopping_patience=None,
            keep_history_on_reset=False,
        ),
    )


def make_cv(
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


def successful_trials(outer) -> list[Any]:
    return [tr for tr in outer.trial_results if tr.status == "SUCCESS"]


# ============================================================
# Constructor / config guards
# ============================================================

def test_nested_cv_rejects_unrebuildable_final_model_configuration():
    model_spec = make_model_spec()
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        )
    )

    with pytest.raises(ValueError, match="unrebuildable"):
        NestedOptunaSearchCV(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
            outer_splitter_cls=StratifiedKFold,
            inner_splitter_cls=StratifiedKFold,
            n_trials=1,
            k_outer=2,
            k_inner=2,
            final_model_dir=None,
            keep_final_model_state_dict_cpu=False,
        )


def test_nested_cv_rejects_invalid_parameter_path(tmp_path):
    model_spec = make_model_spec()
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        )
    )

    with pytest.raises(ValueError, match="must start with 'model/' or 'trainer/'"):
        make_cv(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            outer_splitter_cls=StratifiedKFold,
            inner_splitter_cls=StratifiedKFold,
            parameter_grid={"badpath": ([1.0], "categorical")},
            tmp_path=tmp_path,
        )


# ============================================================
# Full end-to-end logging / auditability
# ============================================================

def test_nested_cv_logs_everything_needed_for_reporting_stratified(tmp_path):
    dataset = TinyClassificationDataset()
    y, _groups = make_labels_and_groups()

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    cv = make_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        parameter_grid={
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([2], "categorical"),
        },
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        random_state=None,
    )

    result = cv.run(dataset, index=y, groups=None)

    assert isinstance(result, NestedCVResult)

    # CV-level metadata
    assert result.outer_splitter_name == "StratifiedKFold"
    assert result.inner_splitter_name == "StratifiedKFold"
    assert result.k_outer == 2
    assert result.k_inner == 2
    assert result.n_trials == 1
    assert result.max_trial_attempts == 3
    assert result.selection_metric_name == "accuracy"
    assert result.selection_metric_direction == "maximize"
    assert result.final_model_dir == str(tmp_path)
    assert result.keep_final_model_state_dict_cpu is True

    assert len(result.outer_results) == 2

    all_dataset_indices = set(range(len(dataset)))

    for outer in result.outer_results:
        # Outer split logging
        assert len(outer.outer_train_indices) > 0
        assert len(outer.outer_test_indices) > 0
        assert set(outer.outer_train_indices).isdisjoint(set(outer.outer_test_indices))
        assert set(outer.outer_train_indices) | set(outer.outer_test_indices) == all_dataset_indices

        # Trial bookkeeping
        assert outer.attempted_trials == 1
        assert outer.successful_trials == 1
        assert outer.failed_trials == 0
        assert outer.pruned_trials == 0
        assert len(outer.trial_results) == 1

        # Best trial / best params
        assert outer.best_trial_number == 0
        assert outer.best_params["model/backbone/kwargs/scale_factor"] == 1.0
        assert outer.best_params["trainer/config/max_epochs"] == 2
        assert outer.best_metric == pytest.approx(1.0)
        assert outer.best_selection_score == pytest.approx(1.0)

        # Selected inner-fold reporting
        assert len(outer.selected_inner_results) == 2
        assert outer.selected_inner_metric_mean == pytest.approx(1.0)
        assert outer.selected_inner_metric_std == pytest.approx(0.0)
        assert outer.selected_inner_metric_min == pytest.approx(1.0)
        assert outer.selected_inner_metric_max == pytest.approx(1.0)

        # Final fit artifacts
        assert outer.final_model_spec is not None
        assert outer.final_trainer_spec is not None
        assert outer.final_fit_epochs is not None 
        assert outer.final_fit_epochs >= 0
        assert outer.final_epochs_ran is not None
        assert outer.final_epochs_ran >= 0

        # Final refit is train-only by design
        assert outer.final_best_epoch is None
        assert outer.final_best_metric is None

        assert len(outer.final_train_logs) >= 1
        assert outer.final_val_logs == []
        assert isinstance(outer.final_history, list)
        assert len(outer.final_history) == 1

        # Final saved model
        assert outer.final_model_state_dict_cpu is not None
        assert outer.final_model_state_dict_path is not None
        assert os.path.exists(outer.final_model_state_dict_path)

        # Holdout metrics
        assert outer.test_metrics is not None
        assert "val/accuracy" in outer.test_metrics
        assert outer.test_metrics["val/accuracy"] == pytest.approx(1.0)

        # Successful trial details
        trial = successful_trials(outer)[0]
        assert trial.params["model/backbone/kwargs/scale_factor"] == 1.0
        assert trial.params["trainer/config/max_epochs"] == 2
        assert trial.aggregate_metric == pytest.approx(1.0)
        assert trial.aggregate_selection_score == pytest.approx(1.0)
        assert trial.error_message is None
        assert trial.error_traceback is None

        # OOF aggregate exact coverage of outer-train, no duplicates, no leakage
        assert sorted(trial.aggregate_oof_sample_indices) == sorted(outer.outer_train_indices)
        assert len(trial.aggregate_oof_sample_indices) == len(set(trial.aggregate_oof_sample_indices))
        assert set(trial.aggregate_oof_sample_indices).isdisjoint(set(outer.outer_test_indices))

        assert "clf" in trial.aggregate_oof_logits
        assert "clf" in trial.aggregate_oof_targets
        assert trial.aggregate_oof_logits["clf"].shape[0] == len(outer.outer_train_indices)
        assert trial.aggregate_oof_targets["clf"].shape[0] == len(outer.outer_train_indices)

        # Each inner fold
        seen_val_indices: set[int] = set()
        for inner in trial.inner_results:
            assert set(inner.inner_train_indices).isdisjoint(set(inner.inner_val_indices))
            assert set(inner.inner_train_indices).issubset(set(outer.outer_train_indices))
            assert set(inner.inner_val_indices).issubset(set(outer.outer_train_indices))

            assert sorted(inner.oof_sample_indices) == sorted(inner.inner_val_indices)

            assert "clf" in inner.oof_logits
            assert "clf" in inner.oof_targets
            assert inner.oof_logits["clf"].shape[0] == len(inner.inner_val_indices)
            assert inner.oof_targets["clf"].shape[0] == len(inner.inner_val_indices)

            seen_val_indices.update(inner.inner_val_indices)

        assert seen_val_indices == set(outer.outer_train_indices)


# ============================================================
# Group leakage checks with real splitters
# ============================================================

@pytest.mark.parametrize(
    "outer_splitter_cls,inner_splitter_cls",
    [
        (GroupKFold, GroupKFold),
        (StratifiedGroupKFold, StratifiedGroupKFold),
    ],
)
def test_nested_cv_group_splitters_have_no_group_leakage(
    tmp_path,
    outer_splitter_cls,
    inner_splitter_cls,
):
    dataset = TinyClassificationDataset()
    y, groups = make_labels_and_groups()

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    cv = make_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        outer_splitter_cls=outer_splitter_cls,
        inner_splitter_cls=inner_splitter_cls,
        parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        random_state=None,
    )

    result = cv.run(dataset, index=y, groups=groups)

    for outer in result.outer_results:
        train_groups = {groups[i] for i in outer.outer_train_indices}
        test_groups = {groups[i] for i in outer.outer_test_indices}
        assert train_groups.isdisjoint(test_groups)

        trial = successful_trials(outer)[0]

        # OOF exact coverage of outer-train only
        assert sorted(trial.aggregate_oof_sample_indices) == sorted(outer.outer_train_indices)
        assert set(trial.aggregate_oof_sample_indices).isdisjoint(set(outer.outer_test_indices))

        for inner in trial.inner_results:
            inner_train_groups = {groups[i] for i in inner.inner_train_indices}
            inner_val_groups = {groups[i] for i in inner.inner_val_indices}
            assert inner_train_groups.isdisjoint(inner_val_groups)


# ============================================================
# Parameter routing and selection direction
# ============================================================

def test_nested_cv_routes_parameters_into_real_model_and_trainer_specs(tmp_path):
    dataset = TinyClassificationDataset()
    y, _groups = make_labels_and_groups()

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    cv = make_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        parameter_grid={
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([4], "categorical"),
        },
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        random_state=None,
    )

    result = cv.run(dataset, index=y, groups=None)

    for outer in result.outer_results:
        assert outer.final_model_spec is not None
        assert outer.final_trainer_spec is not None

        assert outer.final_model_spec.backbone is not None
        assert outer.final_model_spec.backbone.kwargs["scale_factor"] == 1.0
        assert outer.final_trainer_spec.config.max_epochs == 4

        assert outer.best_params["model/backbone/kwargs/scale_factor"] == 1.0
        assert outer.best_params["trainer/config/max_epochs"] == 4

        assert outer.final_trainer_spec.config.max_epochs == 4

        assert outer.final_fit_epochs is not None
        assert outer.final_fit_epochs >= 0

        assert outer.final_epochs_ran is not None
        assert outer.final_epochs_ran >= 0

        assert len(outer.final_train_logs) >= 1


def test_nested_cv_handles_minimize_selection_metric_correctly(tmp_path):
    dataset = TinyClassificationDataset()
    y, _groups = make_labels_and_groups()

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ErrorRateEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
        ),
        max_epochs=2,
    )

    cv = make_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        random_state=None,
    )

    result = cv.run(dataset, index=y, groups=None)

    assert result.selection_metric_name == "error_rate"
    assert result.selection_metric_direction == "minimize"

    for outer in result.outer_results:
        assert outer.best_metric == pytest.approx(0.0)
        assert outer.best_selection_score == pytest.approx(-outer.best_metric)
        assert outer.selected_inner_metric_mean == pytest.approx(0.0)
        assert outer.test_metrics is not None
        assert "val/error_rate" in outer.test_metrics
        assert outer.test_metrics["val/error_rate"] == pytest.approx(0.0)


# ============================================================
# Final calibration and rebuild using real saved state
# ============================================================

def test_nested_cv_rebuilds_final_model_and_trainer_and_preserves_calibrator_fit(tmp_path):
    dataset = TinyClassificationDataset()
    y, _groups = make_labels_and_groups()

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    cv = make_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        random_state=None,
        keep_final_model_state_dict_cpu=False,  # force path-based reconstruction
    )

    result = cv.run(dataset, index=y, groups=None)

    rebuilt_model = result.rebuild_final_model(0, device="cpu")
    assert isinstance(rebuilt_model, TorchkitModel)

    # Calibrator fit should have been persisted into the final saved state_dict
    phead = rebuilt_model.prediction_heads["clf"]
    calibrator = phead.calibrator
    assert calibrator is not None
    assert int(calibrator.fit_calls.item()) == 1

    outer0 = result.outer_results[0]
    assert int(calibrator.last_num_samples.item()) == len(outer0.outer_train_indices)
    assert int(calibrator.last_num_classes.item()) == 2

    # Predict after rebuild should now expose calibrated_logits
    sample = dataset[0]

    batched_sample = {
        "x": sample["x"].unsqueeze(0),
        "y": sample["y"].unsqueeze(0),
    }

    pred = rebuilt_model.predict(
        batched_sample,
        "clf",
        return_raw_head_outputs=True,
    )
    assert "clf" in pred
    assert "logits" in pred["clf"]
    assert "calibrated_logits" in pred["clf"]
    assert pred["clf"]["logits"].shape == (1, 2)
    assert pred["clf"]["calibrated_logits"].shape == (1, 2)

    rebuilt_trainer = result.rebuild_final_trainer(0, device="cpu")
    assert isinstance(rebuilt_trainer, Trainer)
    assert str(rebuilt_trainer.device) == "cpu"


# ============================================================
# Pickle roundtrip is critical for offline analysis
# ============================================================

def test_nested_cv_result_is_pickleable_and_reconstruction_survives_roundtrip(tmp_path):
    dataset = TinyClassificationDataset()
    y, _groups = make_labels_and_groups()

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    cv = make_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        random_state=None,
    )

    result = cv.run(dataset, index=y, groups=None)

    blob = pickle.dumps(result)
    restored = pickle.loads(blob)

    assert isinstance(restored, NestedCVResult)
    assert restored.selection_metric_name == result.selection_metric_name
    assert restored.selection_metric_direction == result.selection_metric_direction
    assert len(restored.outer_results) == len(result.outer_results)

    model = restored.rebuild_final_model(0, device="cpu")
    trainer = restored.rebuild_final_trainer(0, device="cpu")

    assert isinstance(model, TorchkitModel)
    assert isinstance(trainer, Trainer)

    sample = dataset[0]
    pred = model.predict(
        {
            "x": sample["x"].unsqueeze(0),
            "y": sample["y"].unsqueeze(0),
        },
        "clf",
        return_raw_head_outputs=True,
    )
    assert "clf" in pred
    assert "logits" in pred["clf"]
    assert "calibrated_logits" in pred["clf"]

    # Offline analysis should still have all needed information
    outer0 = restored.outer_results[0]
    assert outer0.best_params
    assert outer0.test_metrics is not None
    assert len(outer0.selected_inner_results) > 0
    assert outer0.final_model_spec is not None
    assert outer0.final_trainer_spec is not None


# ============================================================
# Failure / prune bookkeeping using a small fault-injecting subclass
# ============================================================

class FaultInjectingNestedOptunaSearchCV(NestedOptunaSearchCV):
    """
    Real CV class, but forces one pruned and one failed trial before a real success.
    This lets us verify bookkeeping without monkeypatching.
    """
    def _run_single_trial(
        self,
        *,
        trial: optuna.Trial,
        outer_train_dataset,
        outer_train_index,
        outer_train_groups,
        outer_train_original_indices,
    ):
        # Populate trial.params through the real suggest_parameters path first.
        _ = self.suggest_parameters(trial, self.parameter_grid)

        if trial.number == 0:
            raise optuna.TrialPruned("synthetic prune for bookkeeping test")

        if trial.number == 1:
            raise RuntimeError("synthetic failure for bookkeeping test")

        return super()._run_single_trial(
            trial=trial,
            outer_train_dataset=outer_train_dataset,
            outer_train_index=outer_train_index,
            outer_train_groups=outer_train_groups,
            outer_train_original_indices=outer_train_original_indices,
        )


def test_nested_cv_logs_failed_and_pruned_trials_with_params_and_tracebacks(tmp_path):
    dataset = TinyClassificationDataset()
    y, _groups = make_labels_and_groups()

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    cv = FaultInjectingNestedOptunaSearchCV(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        dataloader_factory=lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=2, shuffle=shuffle),
        n_trials=1,
        max_trial_attempts=4,
        k_outer=2,
        k_inner=2,
        shuffle_outer=False,
        shuffle_inner=False,
        random_state=None,
        calibrate=True,
        final_model_dir=str(tmp_path),
        keep_final_model_state_dict_cpu=True,
    )

    result = cv.run(dataset, index=y, groups=None)

    for outer in result.outer_results:
        assert outer.pruned_trials == 1
        assert outer.failed_trials == 1
        assert outer.successful_trials == 1
        assert outer.attempted_trials == 3
        assert len(outer.trial_results) == 3

        pruned = [tr for tr in outer.trial_results if tr.status == "PRUNED"]
        failed = [tr for tr in outer.trial_results if tr.status == "FAILED"]
        success = [tr for tr in outer.trial_results if tr.status == "SUCCESS"]

        assert len(pruned) == 1
        assert len(failed) == 1
        assert len(success) == 1

        # Failed/pruned trials must retain sampled params
        assert pruned[0].params == {"model/backbone/kwargs/scale_factor": 1.0}
        assert failed[0].params == {"model/backbone/kwargs/scale_factor": 1.0}

        assert pruned[0].error_message is not None
        assert failed[0].error_message is not None
        assert pruned[0].error_traceback is not None
        assert failed[0].error_traceback is not None

def test_nested_cv_is_deterministic_for_same_seed(tmp_path):
    dataset = TinyClassificationDataset()
    y, groups = make_labels_and_groups()

    model_spec_1 = make_model_spec(scale_factor=1.0)
    trainer_spec_1 = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    model_spec_2 = make_model_spec(scale_factor=1.0)
    trainer_spec_2 = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    cv1 = make_cv(
        model_spec=model_spec_1,
        trainer_spec=trainer_spec_1,
        outer_splitter_cls=StratifiedGroupKFold,
        inner_splitter_cls=StratifiedGroupKFold,
        parameter_grid={
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([2], "categorical"),
        },
        tmp_path=tmp_path / "run1",
        n_trials=1,
        max_trial_attempts=3,
        k_outer=2,
        k_inner=2,
        random_state=42,
    )

    cv2 = make_cv(
        model_spec=model_spec_2,
        trainer_spec=trainer_spec_2,
        outer_splitter_cls=StratifiedGroupKFold,
        inner_splitter_cls=StratifiedGroupKFold,
        parameter_grid={
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([2], "categorical"),
        },
        tmp_path=tmp_path / "run2",
        n_trials=1,
        max_trial_attempts=3,
        k_outer=2,
        k_inner=2,
        random_state=42,
    )

    result1 = cv1.run(dataset, index=y, groups=groups)
    result2 = cv2.run(dataset, index=y, groups=groups)

    assert len(result1.outer_results) == len(result2.outer_results)

    for outer1, outer2 in zip(result1.outer_results, result2.outer_results):
        assert outer1.outer_train_indices == outer2.outer_train_indices
        assert outer1.outer_test_indices == outer2.outer_test_indices

        assert outer1.best_trial_number == outer2.best_trial_number
        assert outer1.best_params == outer2.best_params
        assert outer1.best_metric == pytest.approx(outer2.best_metric)
        assert outer1.best_selection_score == pytest.approx(outer2.best_selection_score)

        assert outer1.selected_inner_metric_mean == pytest.approx(outer2.selected_inner_metric_mean)
        assert outer1.selected_inner_metric_std == pytest.approx(outer2.selected_inner_metric_std)
        assert outer1.selected_inner_metric_min == pytest.approx(outer2.selected_inner_metric_min)
        assert outer1.selected_inner_metric_max == pytest.approx(outer2.selected_inner_metric_max)

        assert outer1.final_fit_epochs == outer2.final_fit_epochs
        assert outer1.final_epochs_ran == outer2.final_epochs_ran

        assert outer1.test_metrics is not None
        assert outer2.test_metrics is not None
        assert outer1.test_metrics.keys() == outer2.test_metrics.keys()

        for k in outer1.test_metrics:
            v1 = outer1.test_metrics[k]
            v2 = outer2.test_metrics[k]
            if isinstance(v1, float):
                assert v1 == pytest.approx(v2)
            else:
                assert v1 == v2

        successful1 = [tr for tr in outer1.trial_results if tr.status == "SUCCESS"]
        successful2 = [tr for tr in outer2.trial_results if tr.status == "SUCCESS"]

        assert len(successful1) == len(successful2)

        for tr1, tr2 in zip(successful1, successful2):
            assert tr1.params == tr2.params
            assert tr1.aggregate_metric == pytest.approx(tr2.aggregate_metric)
            assert tr1.aggregate_selection_score == pytest.approx(tr2.aggregate_selection_score)
            assert tr1.aggregate_oof_sample_indices == tr2.aggregate_oof_sample_indices

            assert tr1.aggregate_oof_logits.keys() == tr2.aggregate_oof_logits.keys()
            for task in tr1.aggregate_oof_logits:
                assert torch.equal(tr1.aggregate_oof_logits[task], tr2.aggregate_oof_logits[task])
                assert torch.equal(tr1.aggregate_oof_targets[task], tr2.aggregate_oof_targets[task])

            assert len(tr1.inner_results) == len(tr2.inner_results)
            for in1, in2 in zip(tr1.inner_results, tr2.inner_results):
                assert in1.inner_train_indices == in2.inner_train_indices
                assert in1.inner_val_indices == in2.inner_val_indices
                assert in1.best_metric == pytest.approx(in2.best_metric)
                assert in1.best_epoch == in2.best_epoch
                assert in1.oof_sample_indices == in2.oof_sample_indices

                assert in1.oof_logits.keys() == in2.oof_logits.keys()
                for task in in1.oof_logits:
                    assert torch.equal(in1.oof_logits[task], in2.oof_logits[task])
                    assert torch.equal(in1.oof_targets[task], in2.oof_targets[task])
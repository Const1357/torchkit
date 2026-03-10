from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from torchkit.models.Model._model import TorchkitModel
from torchkit.models.backbone._backbone import Backbone
from torchkit.models.head._task_head import TaskHead
from torchkit.models.prediction._prediction_head import PredictionHead

from torchkit.models.fuse._fuse_module import FuseModule
from torchkit.models.adapters._feature_adapter import FeatureAdapter
from torchkit.models.calibration._calibrator import Calibrator
from torchkit.models.probability_mapping._probability_mapper import ProbabilityMapper
from torchkit.models.decision._decision_module import DecisionModule

from torchkit.evaluate.classification_evaluator import ClassificationEvaluator
from torchkit.evaluate.regression_evaluator import RegressionEvaluator
from torchkit.evaluate.calibration_evaluator import CalibrationEvaluator
from torchkit.evaluate.roc_evaluator import ROCBinaryEvaluator
from torchkit.evaluate.dca_evaluator import DCAEvaluator
from torchkit.evaluate._evaluator import CompositeEvaluator


# ============================================================
# Dummy model components
# ============================================================

class DummyBackbone(Backbone):
    def __init__(self):
        super().__init__(supported_features=["feat_a", "feat_b"])

    def _forward_impl(
        self,
        input: dict[str, Tensor],
        *,
        requested_features=None,
        **kwargs,
    ) -> dict[str, Tensor]:
        x = input["x"]
        out = {}
        if "feat_a" in requested_features:
            out["feat_a"] = x + 1.0
        if "feat_b" in requested_features:
            out["feat_b"] = x + 2.0
        return out


class DummyFuse(FuseModule):
    def forward(self, features: dict[str, Tensor], **kwargs) -> Tensor:
        return torch.cat([features[k] for k in sorted(features.keys())], dim=1)


class IdentityAdapter(FeatureAdapter):
    def forward(self, features: Tensor, **kwargs) -> Tensor:
        return features


class LinearLogitsHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        return {"logits": self.linear(x)}


class LinearPredictionsHead(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, **kwargs) -> dict[str, Tensor]:
        return {"predictions": self.linear(x)}


class AdditiveCalibrator(Calibrator):
    def __init__(self, add_value: float = 0.5, *, active: bool = False):
        super().__init__(active=active)
        self.add_value = float(add_value)

    def fit_impl(self, logits: Tensor, targets: Tensor) -> None:
        return None

    def forward_impl(self, logits: Tensor) -> Tensor:
        return logits + self.add_value


class SoftmaxProbabilityMapper(ProbabilityMapper):
    def forward_impl(self, logits: Tensor) -> Tensor:
        if logits.ndim == 1:
            return torch.sigmoid(logits)
        if logits.ndim == 2 and logits.shape[1] == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)


class ArgmaxDecisionModule(DecisionModule):
    def forward_impl(self, probs: Tensor) -> Tensor:
        if probs.ndim == 1:
            return (probs >= 0.5).long()
        if probs.ndim == 2 and probs.shape[1] == 1:
            return (probs[:, 0] >= 0.5).long()
        return torch.argmax(probs, dim=1)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def x_tensor() -> Tensor:
    # shape: (N=6, D=3)
    return torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [0.1, 1.2, -0.4],
            [-0.3, 0.7, 1.5],
            [1.1, -0.2, 0.0],
            [0.3, 0.6, -1.2],
            [-0.7, 0.8, 0.4],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def batch_payload(x_tensor: Tensor) -> dict[str, Tensor]:
    return {
        "x": x_tensor.clone(),
        "y_clf": torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.long),
        "y_reg": torch.tensor([[1.0], [2.0], [3.0], [1.5], [2.5], [0.5]], dtype=torch.float32),
    }


@pytest.fixture
def model_with_prediction_head() -> TorchkitModel:
    backbone = DummyBackbone()

    clf_head = TaskHead(
        required_features="feat_a",
        feature_adapter=IdentityAdapter(),
        head_module=LinearLogitsHead(in_features=3, out_features=2),
    )

    phead = PredictionHead(
        calibrator=AdditiveCalibrator(add_value=0.25, active=False),
        probability_mapper=SoftmaxProbabilityMapper(),
        decision_module=ArgmaxDecisionModule(),
        active=True,
    )

    model = TorchkitModel(
        backbone=backbone,
        heads={"clf": clf_head},
        prediction_heads={"clf": phead},
    )
    model.eval()
    return model


@pytest.fixture
def multitask_model_mixed_prediction_heads() -> TorchkitModel:
    backbone = DummyBackbone()

    clf_head = TaskHead(
        required_features="feat_a",
        feature_adapter=IdentityAdapter(),
        head_module=LinearLogitsHead(in_features=3, out_features=2),
    )

    reg_head = TaskHead(
        required_features={"feat_a", "feat_b"},
        fuse_module=DummyFuse(),
        feature_adapter=IdentityAdapter(),
        head_module=LinearPredictionsHead(in_features=6, out_features=1),
    )

    clf_phead = PredictionHead(
        calibrator=AdditiveCalibrator(add_value=0.5, active=False),
        probability_mapper=SoftmaxProbabilityMapper(),
        decision_module=ArgmaxDecisionModule(),
        active=True,
    )

    model = TorchkitModel(
        backbone=backbone,
        heads={"clf": clf_head, "reg": reg_head},
        prediction_heads={"clf": clf_phead},
    )
    model.eval()
    return model


# ============================================================
# Integration tests: model.predict -> evaluator
# ============================================================

def test_model_predict_outputs_work_with_classification_evaluator(
    model_with_prediction_head: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    pred_out = model_with_prediction_head.predict(
        batch_payload,
        "clf",
        return_raw_head_outputs=True,
    )

    eval_inputs = {
        "clf": pred_out["clf"],
        "batch": {"y": batch_payload["y_clf"]},
    }

    evaluator = ClassificationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        predictions_key="clf/predictions",
    )

    metrics = evaluator(inputs=eval_inputs)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "confusion_matrix" in metrics


def test_model_predict_outputs_work_with_calibration_evaluator(
    model_with_prediction_head: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    pred_out = model_with_prediction_head.predict(
        batch_payload,
        "clf",
        return_raw_head_outputs=True,
    )

    eval_inputs = {
        "clf": pred_out["clf"],
        "batch": {"y": batch_payload["y_clf"]},
    }

    evaluator = CalibrationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
    )

    metrics = evaluator(inputs=eval_inputs)

    assert "brier" in metrics
    assert "ece" in metrics
    assert "mce" in metrics


def test_model_predict_outputs_work_with_roc_evaluator(
    model_with_prediction_head: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    pred_out = model_with_prediction_head.predict(
        batch_payload,
        "clf",
        return_raw_head_outputs=True,
    )

    eval_inputs = {
        "clf": pred_out["clf"],
        "batch": {"y": batch_payload["y_clf"]},
    }

    evaluator = ROCBinaryEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
    )

    metrics = evaluator(inputs=eval_inputs)

    assert "auc" in metrics
    assert "roc_curve" in metrics
    assert "youden_threshold" in metrics


def test_model_predict_outputs_work_with_dca_evaluator(
    model_with_prediction_head: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    pred_out = model_with_prediction_head.predict(
        batch_payload,
        "clf",
        return_raw_head_outputs=True,
    )

    eval_inputs = {
        "clf": pred_out["clf"],
        "batch": {"y": batch_payload["y_clf"]},
    }

    evaluator = DCAEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        n_thresholds=20,
    )

    metrics = evaluator(inputs=eval_inputs)

    assert "max_net_benefit" in metrics
    assert "best_threshold" in metrics
    assert "dca_curve" in metrics


def test_model_predict_outputs_work_with_regression_evaluator(
    multitask_model_mixed_prediction_heads: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    pred_out = multitask_model_mixed_prediction_heads.predict(
        batch_payload,
        "reg",
        return_raw_head_outputs=True,
    )

    eval_inputs = {
        "reg": pred_out["reg"],
        "batch": {"target": batch_payload["y_reg"]},
    }

    evaluator = RegressionEvaluator(
        pred_key="reg/predictions",
        target_key="batch/target",
    )

    metrics = evaluator(inputs=eval_inputs)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics


def test_model_predict_outputs_work_with_composite_evaluator(
    model_with_prediction_head: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    pred_out = model_with_prediction_head.predict(
        batch_payload,
        "clf",
        return_raw_head_outputs=True,
    )

    eval_inputs = {
        "clf": pred_out["clf"],
        "batch": {"y": batch_payload["y_clf"]},
    }

    clf_eval = ClassificationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        predictions_key="clf/predictions",
        name="classification",
    )

    cal_eval = CalibrationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        name="calibration",
    )

    composite = CompositeEvaluator(
        [clf_eval, cal_eval],
        name="composite",
        primary_metric="__primary__",
    )

    metrics = composite(inputs=eval_inputs)

    assert "classification/accuracy" in metrics
    assert "classification/macro_f1" in metrics
    assert "calibration/brier" in metrics
    assert "__primary__" in metrics


def test_calibrated_logits_appear_only_when_calibrator_enabled(
    model_with_prediction_head: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    pred_disabled = model_with_prediction_head.predict(
        batch_payload,
        "clf",
        return_raw_head_outputs=True,
    )
    assert "logits" in pred_disabled["clf"]
    assert "probabilities" in pred_disabled["clf"]
    assert "predictions" in pred_disabled["clf"]
    assert "calibrated_logits" not in pred_disabled["clf"]

    model_with_prediction_head.prediction_heads["clf"].calibrator.enable()

    pred_enabled = model_with_prediction_head.predict(
        batch_payload,
        "clf",
        return_raw_head_outputs=True,
    )
    assert "calibrated_logits" in pred_enabled["clf"]

    evaluator = ClassificationEvaluator(
        score_key="clf/calibrated_logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        predictions_key="clf/predictions",
    )

    metrics = evaluator(
        inputs={
            "clf": pred_enabled["clf"],
            "batch": {"y": batch_payload["y_clf"]},
        }
    )
    assert "accuracy" in metrics


def test_classification_evaluator_can_fallback_from_calibrated_logits_when_disabled(
    model_with_prediction_head: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    # calibrator remains disabled -> no calibrated_logits in predict output
    pred_out = model_with_prediction_head.predict(
        batch_payload,
        "clf",
        return_raw_head_outputs=True,
    )
    assert "calibrated_logits" not in pred_out["clf"]

    evaluator = ClassificationEvaluator(
        score_key="clf/calibrated_logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        predictions_key="clf/predictions",
    )

    metrics = evaluator(
        inputs={
            "clf": pred_out["clf"],
            "batch": {"y": batch_payload["y_clf"]},
        }
    )
    assert "accuracy" in metrics


def test_multitask_predict_mixed_prediction_heads_can_be_evaluated_independently(
    multitask_model_mixed_prediction_heads: TorchkitModel,
    batch_payload: dict[str, Tensor],
):
    pred_out = multitask_model_mixed_prediction_heads.predict(
        batch_payload,
        "clf",
        "reg",
        return_raw_head_outputs=True,
    )

    clf_metrics = ClassificationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        predictions_key="clf/predictions",
    )(
        inputs={
            "clf": pred_out["clf"],
            "batch": {"y": batch_payload["y_clf"]},
        }
    )

    reg_metrics = RegressionEvaluator(
        pred_key="reg/predictions",
        target_key="batch/target",
    )(
        inputs={
            "reg": pred_out["reg"],
            "batch": {"target": batch_payload["y_reg"]},
        }
    )

    assert "accuracy" in clf_metrics
    assert "rmse" in reg_metrics
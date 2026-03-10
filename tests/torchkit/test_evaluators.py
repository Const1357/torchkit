from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from torchkit.evaluate._evaluator import Evaluator, CompositeEvaluator
from torchkit.evaluate.classification_evaluator import ClassificationEvaluator
from torchkit.evaluate.calibration_evaluator import CalibrationEvaluator
from torchkit.evaluate.regression_evaluator import RegressionEvaluator
from torchkit.evaluate.roc_evaluator import ROCBinaryEvaluator
from torchkit.evaluate.dca_evaluator import DCAEvaluator
from torchkit.evaluate.segmentation_evaluator import (
    SegmentationEvaluator,
    Segmentation3DEvaluator,
)


# ============================================================
# Dummy evaluators for base tests
# ============================================================

class DummyEvaluator(Evaluator):
    def __init__(self):
        super().__init__(
            name="dummy",
            primary_metric="score",
            direction="maximize",
            weight=1.0,
        )

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("a/b",)

    def metrics(self, *, inputs: dict[str, object]) -> dict[str, object]:
        x = self.resolve(inputs, "a/b")
        return {"score": float(x.mean())}


class OptionalDummyEvaluator(Evaluator):
    def __init__(self):
        super().__init__(
            name="optional_dummy",
            primary_metric="score",
            direction="maximize",
            weight=1.0,
        )

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("x",)

    @property
    def optional_keys(self) -> tuple[str, ...]:
        return ("maybe",)

    def metrics(self, *, inputs: dict[str, object]) -> dict[str, object]:
        x = self.resolve(inputs, "x")
        maybe = self.resolve(inputs, "maybe", strict=False)
        bonus = 0.0 if maybe is None else float(maybe.mean())
        return {"score": float(x.mean()) + bonus}


class BadMetricsEvaluator(Evaluator):
    def __init__(self):
        super().__init__(
            name="bad",
            primary_metric="score",
            direction="maximize",
            weight=1.0,
        )

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("x",)

    def metrics(self, *, inputs: dict[str, object]):
        return "not a dict"


class MissingPrimaryEvaluator(Evaluator):
    def __init__(self):
        super().__init__(
            name="missing_primary",
            primary_metric="score",
            direction="maximize",
            weight=1.0,
        )

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("x",)

    def metrics(self, *, inputs: dict[str, object]) -> dict[str, object]:
        return {"other": 1.0}


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def nested_inputs() -> dict[str, object]:
    return {
        "a": {"b": torch.tensor([1.0, 2.0, 3.0])},
        "x": torch.tensor([1.0, 2.0]),
        "maybe": None,
    }


@pytest.fixture
def classification_inputs_multiclass() -> dict[str, object]:
    return {
        "clf": {
            "logits": torch.tensor(
                [
                    [4.0, 1.0, 0.0],
                    [0.0, 4.0, 1.0],
                    [0.0, 1.0, 4.0],
                    [3.0, 2.0, 1.0],
                ],
                dtype=torch.float32,
            )
        },
        "batch": {
            "y": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        },
    }


@pytest.fixture
def classification_inputs_binary_n2() -> dict[str, object]:
    logits = torch.tensor(
        [
            [-2.0, 2.0],
            [2.0, -2.0],
            [-1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    return {
        "clf": {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        },
        "batch": {
            "y": torch.tensor([1, 0, 1, 0], dtype=torch.long),
        },
    }


@pytest.fixture
def classification_inputs_binary_n1() -> dict[str, object]:
    logits = torch.tensor([[2.0], [-2.0], [1.0], [-1.0]], dtype=torch.float32)
    probs_pos = torch.sigmoid(logits[:, 0])
    preds = (probs_pos >= 0.5).long()

    return {
        "clf": {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "predictions": preds,
        },
        "batch": {
            "y": torch.tensor([1, 0, 1, 0], dtype=torch.long),
        },
    }


@pytest.fixture
def classification_inputs_binary_n() -> dict[str, object]:
    logits = torch.tensor([2.0, -2.0, 1.0, -1.0], dtype=torch.float32)
    probs_pos = torch.sigmoid(logits)
    preds = (probs_pos >= 0.5).long()

    return {
        "clf": {
            "logits": logits,
            "probabilities": probs_pos,
            "predictions": preds,
        },
        "batch": {
            "y": torch.tensor([1, 0, 1, 0], dtype=torch.long),
        },
    }


@pytest.fixture
def calibration_inputs() -> dict[str, object]:
    logits = torch.tensor(
        [
            [-2.0, 2.0],
            [2.0, -2.0],
            [-0.5, 0.5],
            [0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    probs = torch.softmax(logits, dim=1)

    return {
        "clf": {
            "logits": logits,
            "probabilities": probs,
        },
        "batch": {
            "y": torch.tensor([1, 0, 1, 0], dtype=torch.long),
        },
    }


@pytest.fixture
def regression_inputs() -> dict[str, object]:
    return {
        "reg": {
            "predictions": torch.tensor(
                [
                    [1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                ],
                dtype=torch.float32,
            )
        },
        "batch": {
            "target": torch.tensor(
                [
                    [1.5, 2.5],
                    [2.5, 2.5],
                    [2.5, 3.5],
                ],
                dtype=torch.float32,
            )
        },
    }


@pytest.fixture
def seg2d_binary_inputs() -> dict[str, object]:
    logits = torch.tensor(
        [
            [[[3.0, -3.0], [3.0, -3.0]]],
            [[[2.0, 2.0], [-2.0, -2.0]]],
        ],
        dtype=torch.float32,
    )  # (B=2,C=1,H=2,W=2)

    targets = torch.tensor(
        [
            [[1, 0], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=torch.long,
    )  # (B=2,H=2,W=2)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long().squeeze(1)

    return {
        "seg": {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        },
        "batch": {
            "mask": targets,
        },
    }


@pytest.fixture
def seg2d_multiclass_inputs() -> dict[str, object]:
    logits = torch.tensor(
        [
            [
                [[4.0, -2.0], [-2.0, -2.0]],
                [[-2.0, 4.0], [-2.0, -2.0]],
                [[-2.0, -2.0], [4.0, 4.0]],
            ]
        ],
        dtype=torch.float32,
    )  # (1,3,2,2)

    targets = torch.tensor(
        [[[0, 1], [2, 2]]],
        dtype=torch.long,
    )  # (1,2,2)

    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)

    return {
        "seg": {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        },
        "batch": {
            "mask": targets,
        },
    }


@pytest.fixture
def seg3d_binary_inputs() -> dict[str, object]:
    logits = torch.tensor(
        [
            [[
                [[3.0, -3.0], [3.0, -3.0]],
                [[3.0, -3.0], [3.0, -3.0]],
            ]]
        ],
        dtype=torch.float32,
    )  # (1,1,2,2,2)

    targets = torch.tensor(
        [
            [
                [[1, 0], [1, 0]],
                [[1, 0], [1, 0]],
            ]
        ],
        dtype=torch.long,
    )  # (1,2,2,2)

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long().squeeze(1)

    return {
        "seg": {
            "logits": logits,
            "probabilities": probs,
            "predictions": preds,
        },
        "batch": {
            "mask": targets,
        },
    }


# ============================================================
# Base Evaluator tests
# ============================================================

def test_evaluator_resolve_success(nested_inputs: dict[str, object]):
    x = Evaluator.resolve(nested_inputs, "a/b")
    assert torch.equal(x, torch.tensor([1.0, 2.0, 3.0]))


def test_evaluator_resolve_missing_key_raises(nested_inputs: dict[str, object]):
    with pytest.raises(KeyError, match="Key 'missing' not found"):
        Evaluator.resolve(nested_inputs, "a/missing")


def test_evaluator_resolve_non_dict_midpath_raises(nested_inputs: dict[str, object]):
    with pytest.raises(TypeError, match="Expected dict at path a/b"):
        Evaluator.resolve(nested_inputs, "a/b/c")


def test_evaluator_resolve_nontensor_raises():
    with pytest.raises(TypeError, match="must be Tensor"):
        Evaluator.resolve({"a": {"b": 123}}, "a/b")


def test_evaluator_resolve_strict_false_allows_none():
    out = Evaluator.resolve({"a": {"b": None}}, "a/b", strict=False)
    assert out is None


def test_evaluator_call_runs_metrics(nested_inputs: dict[str, object]):
    ev = DummyEvaluator()
    metrics = ev(inputs=nested_inputs)

    assert isinstance(metrics, dict)
    assert "score" in metrics
    assert metrics["score"] == 2.0


def test_evaluator_call_rejects_non_dict_input():
    ev = DummyEvaluator()

    with pytest.raises(TypeError, match="inputs must be dict"):
        ev(inputs=[1, 2, 3])  # type: ignore[arg-type]


def test_evaluator_call_rejects_missing_required_keys():
    ev = DummyEvaluator()

    with pytest.raises(KeyError, match="missing required keys"):
        ev(inputs={"a": {}})


def test_evaluator_optional_keys_may_be_none(nested_inputs: dict[str, object]):
    ev = OptionalDummyEvaluator()
    metrics = ev(inputs=nested_inputs)

    assert "score" in metrics
    assert math.isfinite(metrics["score"])


def test_evaluator_rejects_non_dict_metrics():
    ev = BadMetricsEvaluator()

    with pytest.raises(TypeError, match="must return dict"):
        ev(inputs={"x": torch.tensor([1.0])})


def test_evaluator_rejects_missing_primary_metric():
    ev = MissingPrimaryEvaluator()

    with pytest.raises(KeyError, match="Primary metric 'score' not found"):
        ev(inputs={"x": torch.tensor([1.0])})


def test_evaluator_primary_value():
    ev = DummyEvaluator()
    assert ev.primary_value(metrics={"score": 1.5}) == 1.5
    assert ev.primary_value(metrics={"score": None}) is None


# ============================================================
# CompositeEvaluator tests
# ============================================================

def test_composite_evaluator_requires_nonempty():
    with pytest.raises(ValueError, match="requires a non-empty"):
        CompositeEvaluator([])


def test_composite_evaluator_rejects_duplicate_names():
    ev1 = DummyEvaluator()
    ev2 = DummyEvaluator()

    with pytest.raises(ValueError, match="must be unique"):
        CompositeEvaluator([ev1, ev2])


def test_composite_evaluator_namespaces_metrics(nested_inputs: dict[str, object]):
    ev1 = DummyEvaluator()
    ev2 = OptionalDummyEvaluator()

    comp = CompositeEvaluator([ev1, ev2], name="comp", primary_metric="__primary__")
    metrics = comp(inputs=nested_inputs)

    assert "dummy/score" in metrics
    assert "optional_dummy/score" in metrics
    assert "__primary__" in metrics


def test_composite_evaluator_primary_combines_weights(nested_inputs: dict[str, object]):
    ev1 = DummyEvaluator()
    ev2 = OptionalDummyEvaluator()

    comp = CompositeEvaluator([ev1, ev2], name="comp", primary_metric="__primary__")
    metrics = comp(inputs=nested_inputs)

    expected = metrics["dummy/score"] + metrics["optional_dummy/score"]
    assert metrics["__primary__"] == expected


# ============================================================
# ClassificationEvaluator tests
# ============================================================

def test_classification_evaluator_multiclass(classification_inputs_multiclass: dict[str, object]):
    ev = ClassificationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
    )

    metrics = ev(inputs=classification_inputs_multiclass)

    assert "macro_f1" in metrics
    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 3
    assert math.isfinite(metrics["macro_f1"])


@pytest.mark.parametrize(
    "fixture_name",
    ["classification_inputs_binary_n2", "classification_inputs_binary_n1", "classification_inputs_binary_n"],
)
def test_classification_evaluator_binary_supported_shapes(request: pytest.FixtureRequest, fixture_name: str):
    inputs = request.getfixturevalue(fixture_name)

    ev = ClassificationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
    )
    metrics = ev(inputs=inputs)

    assert "accuracy" in metrics
    assert "pr_auc" in metrics
    assert "pr_curve" in metrics
    assert len(metrics["confusion_matrix"]) == 2


def test_classification_evaluator_uses_explicit_probabilities_and_predictions(
    classification_inputs_binary_n2: dict[str, object],
):
    ev = ClassificationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        predictions_key="clf/predictions",
    )

    metrics = ev(inputs=classification_inputs_binary_n2)

    assert "accuracy" in metrics
    assert metrics["accuracy"] == 1.0


def test_classification_evaluator_fallback_from_calibrated_logits(
    classification_inputs_binary_n2: dict[str, object],
):
    ev = ClassificationEvaluator(
        score_key="clf/calibrated_logits",
        target_key="batch/y",
    )

    metrics = ev(inputs=classification_inputs_binary_n2)
    assert "accuracy" in metrics


# ============================================================
# CalibrationEvaluator tests
# ============================================================

def test_calibration_evaluator_runs(calibration_inputs: dict[str, object]):
    ev = CalibrationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
    )

    metrics = ev(inputs=calibration_inputs)

    assert "brier" in metrics
    assert "ece" in metrics
    assert "mce" in metrics
    assert "calibration_curve" in metrics
    assert math.isfinite(metrics["brier"])


def test_calibration_evaluator_uses_explicit_probabilities(calibration_inputs: dict[str, object]):
    ev = CalibrationEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
    )

    metrics = ev(inputs=calibration_inputs)
    assert "brier" in metrics
    assert math.isfinite(metrics["ece"])


def test_calibration_evaluator_fallback_from_calibrated_logits(
    calibration_inputs: dict[str, object],
):
    ev = CalibrationEvaluator(
        score_key="clf/calibrated_logits",
        target_key="batch/y",
    )

    metrics = ev(inputs=calibration_inputs)
    assert "brier" in metrics


# ============================================================
# RegressionEvaluator tests
# ============================================================

def test_regression_evaluator_runs(regression_inputs: dict[str, object]):
    ev = RegressionEvaluator(
        pred_key="reg/predictions",
        target_key="batch/target",
    )

    metrics = ev(inputs=regression_inputs)

    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert "pearson" in metrics
    assert "rmse/target_0" in metrics
    assert "rmse/target_1" in metrics


def test_regression_evaluator_accepts_1d():
    inputs = {
        "reg": {"predictions": torch.tensor([1.0, 2.0, 3.0])},
        "batch": {"target": torch.tensor([1.5, 2.5, 2.0])},
    }

    ev = RegressionEvaluator(
        pred_key="reg/predictions",
        target_key="batch/target",
    )

    metrics = ev(inputs=inputs)
    assert "rmse" in metrics


def test_regression_evaluator_rejects_shape_mismatch():
    inputs = {
        "reg": {"predictions": torch.tensor([[1.0], [2.0]])},
        "batch": {"target": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
    }

    ev = RegressionEvaluator(
        pred_key="reg/predictions",
        target_key="batch/target",
    )

    with pytest.raises(ValueError, match="identical shape"):
        ev(inputs=inputs)


# ============================================================
# ROCBinaryEvaluator tests
# ============================================================

@pytest.mark.parametrize(
    "fixture_name",
    ["classification_inputs_binary_n2", "classification_inputs_binary_n1", "classification_inputs_binary_n"],
)
def test_roc_binary_evaluator_supported_shapes(request: pytest.FixtureRequest, fixture_name: str):
    inputs = request.getfixturevalue(fixture_name)

    ev = ROCBinaryEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
    )

    metrics = ev(inputs=inputs)

    assert "auc" in metrics
    assert "roc_curve" in metrics
    assert "youden_threshold" in metrics
    assert math.isfinite(metrics["auc"])


def test_roc_binary_evaluator_uses_probabilities(classification_inputs_binary_n2: dict[str, object]):
    ev = ROCBinaryEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
    )

    metrics = ev(inputs=classification_inputs_binary_n2)
    assert "auc" in metrics


def test_roc_binary_evaluator_rejects_single_class():
    inputs = {
        "clf": {"logits": torch.tensor([2.0, 1.0, 3.0])},
        "batch": {"y": torch.tensor([1, 1, 1], dtype=torch.long)},
    }

    ev = ROCBinaryEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
    )

    with pytest.raises(ValueError, match="requires both positive and negative samples"):
        ev(inputs=inputs)

def test_roc_binary_evaluator_fallback_from_calibrated_logits(
    classification_inputs_binary_n2: dict[str, object],
):
    ev = ROCBinaryEvaluator(
        score_key="clf/calibrated_logits",
        target_key="batch/y",
    )

    metrics = ev(inputs=classification_inputs_binary_n2)
    assert "auc" in metrics


# ============================================================
# DCAEvaluator tests
# ============================================================

@pytest.mark.parametrize(
    "fixture_name",
    ["classification_inputs_binary_n2", "classification_inputs_binary_n1", "classification_inputs_binary_n"],
)
def test_dca_evaluator_supported_shapes(request: pytest.FixtureRequest, fixture_name: str):
    inputs = request.getfixturevalue(fixture_name)

    ev = DCAEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        n_thresholds=25,
    )

    metrics = ev(inputs=inputs)

    assert "max_net_benefit" in metrics
    assert "best_threshold" in metrics
    assert "dca_curve" in metrics
    assert len(metrics["dca_curve"]["thresholds"]) == 25


def test_dca_evaluator_uses_probabilities(classification_inputs_binary_n2: dict[str, object]):
    ev = DCAEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        n_thresholds=10,
    )

    metrics = ev(inputs=classification_inputs_binary_n2)
    assert "max_net_benefit" in metrics

def test_dca_evaluator_fallback_from_calibrated_logits(
    classification_inputs_binary_n2: dict[str, object],
):
    ev = DCAEvaluator(
        score_key="clf/calibrated_logits",
        target_key="batch/y",
        n_thresholds=10,
    )

    metrics = ev(inputs=classification_inputs_binary_n2)
    assert "max_net_benefit" in metrics


# ============================================================
# SegmentationEvaluator tests
# ============================================================

def test_segmentation_evaluator_binary_from_scores(seg2d_binary_inputs: dict[str, object]):
    ev = SegmentationEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
    )

    metrics = ev(inputs=seg2d_binary_inputs)

    assert "dice" in metrics
    assert "iou" in metrics
    assert "pixel_accuracy" in metrics
    assert "dice/class_0" in metrics
    assert "dice/class_1" in metrics


def test_segmentation_evaluator_binary_from_probabilities(seg2d_binary_inputs: dict[str, object]):
    ev = SegmentationEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        probabilities_key="seg/probabilities",
    )

    metrics = ev(inputs=seg2d_binary_inputs)
    assert "dice" in metrics


def test_segmentation_evaluator_binary_from_predictions(seg2d_binary_inputs: dict[str, object]):
    ev = SegmentationEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        predictions_key="seg/predictions",
    )

    metrics = ev(inputs=seg2d_binary_inputs)
    assert "dice" in metrics


def test_segmentation_evaluator_multiclass(seg2d_multiclass_inputs: dict[str, object]):
    ev = SegmentationEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
    )

    metrics = ev(inputs=seg2d_multiclass_inputs)

    assert "dice" in metrics
    assert "dice/class_0" in metrics
    assert "dice/class_1" in metrics
    assert "dice/class_2" in metrics

def test_segmentation_evaluator_fallback_from_calibrated_logits(
    seg2d_multiclass_inputs: dict[str, object],
):
    ev = SegmentationEvaluator(
        score_key="seg/calibrated_logits",
        target_key="batch/mask",
    )

    metrics = ev(inputs=seg2d_multiclass_inputs)
    assert "dice" in metrics


# ============================================================
# Segmentation3DEvaluator tests
# ============================================================

def test_segmentation3d_evaluator_returns_none_metrics_when_target_missing(seg3d_binary_inputs: dict[str, object]):
    ev = Segmentation3DEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        include_background=True,
    )

    inputs = {
        "seg": seg3d_binary_inputs["seg"],
        "batch": {"mask": None},
    }

    metrics = ev(inputs=inputs)

    assert metrics["dice"] is None
    assert metrics["iou"] is None
    assert metrics["voxel_accuracy"] is None


def test_segmentation3d_evaluator_binary_from_scores(seg3d_binary_inputs: dict[str, object]):
    ev = Segmentation3DEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        include_background=True,
    )

    metrics = ev(inputs=seg3d_binary_inputs)

    assert "dice" in metrics
    assert "iou" in metrics
    assert "voxel_accuracy" in metrics
    assert "dice/class_0" in metrics
    assert "dice/class_1" in metrics


def test_segmentation3d_evaluator_binary_from_probabilities(seg3d_binary_inputs: dict[str, object]):
    ev = Segmentation3DEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        probabilities_key="seg/probabilities",
        include_background=True,
    )

    metrics = ev(inputs=seg3d_binary_inputs)
    assert "dice" in metrics


def test_segmentation3d_evaluator_binary_from_predictions(seg3d_binary_inputs: dict[str, object]):
    ev = Segmentation3DEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        predictions_key="seg/predictions",
        include_background=True,
    )

    metrics = ev(inputs=seg3d_binary_inputs)
    assert "dice" in metrics


def test_segmentation3d_evaluator_excludes_background(seg3d_binary_inputs: dict[str, object]):
    ev = Segmentation3DEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        include_background=False,
    )

    metrics = ev(inputs=seg3d_binary_inputs)

    assert "dice/class_1" in metrics
    assert "dice/class_0" not in metrics


def test_segmentation3d_evaluator_rejects_wrong_target_shape(seg3d_binary_inputs: dict[str, object]):
    ev = Segmentation3DEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
    )

    inputs = {
        "seg": seg3d_binary_inputs["seg"],
        "batch": {"mask": torch.randn(1, 1, 2, 2, 2)},
    }

    with pytest.raises(ValueError, match="targets must be"):
        ev(inputs=inputs)

def test_segmentation3d_evaluator_fallback_from_calibrated_logits(
    seg3d_binary_inputs: dict[str, object],
):
    ev = Segmentation3DEvaluator(
        score_key="seg/calibrated_logits",
        target_key="batch/mask",
        include_background=True,
    )

    metrics = ev(inputs=seg3d_binary_inputs)
    assert "dice" in metrics
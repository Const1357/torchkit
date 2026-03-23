from __future__ import annotations

import math

import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from torchkit.evaluate._evaluator import Evaluator
from torchkit.evaluate.report import (
    CalibrationReportEvaluator,
    ClassificationReportEvaluator,
    CompositeReportEvaluator,
    DCAReportEvaluator,
    ROCBinaryReportEvaluator,
    RegressionReportEvaluator,
    Segmentation3DReportEvaluator,
    SegmentationReportEvaluator,
)
from torchkit.evaluate.report._report_evaluator import ReportEvaluator
from torchkit.evaluate.select import (
    AccuracySelectorEvaluator,
    BrierScoreSelectorEvaluator,
    BinaryPRAUCSelectorEvaluator,
    CompositeSelectorEvaluator,
    MaximumNetBenefitSelectorEvaluator,
    MeanSquaredErrorSelectorEvaluator,
    ROCAUCSelectorEvaluator,
    Segmentation3DDiceSelectorEvaluator,
    SegmentationDiceSelectorEvaluator,
)
from torchkit.evaluate.select._selector_evaluator import SelectorEvaluator


class DummyReportEvaluator(ReportEvaluator):
    def __init__(self, name: str = "dummy_report") -> None:
        super().__init__(name=name)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("x",)

    def metrics(self, *, inputs: dict[str, object]) -> dict[str, object]:
        x = self.resolve(inputs, "x")
        return {"mean": float(x.mean())}


class DummySelectorEvaluator(SelectorEvaluator):
    def __init__(self, *, name: str = "dummy_selector", direction: str = "maximize", weight: float = 1.0) -> None:
        super().__init__(name=name, direction=direction, weight=weight)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return ("x",)

    def primary_metric(self, *, inputs: dict[str, object]) -> torch.Tensor:
        x = self.resolve(inputs, "x")
        return x.float().mean()


def _binary_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def _ece_and_mce(probs: torch.Tensor, targets: torch.Tensor, n_bins: int) -> tuple[float, float]:
    bin_edges = torch.linspace(0, 1, n_bins + 1, dtype=probs.dtype)
    ece = torch.tensor(0.0, dtype=probs.dtype)
    mce = torch.tensor(0.0, dtype=probs.dtype)
    n = len(probs)
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        conf = probs[mask].mean()
        acc = targets[mask].float().mean()
        gap = torch.abs(acc - conf)
        ece += gap * (mask.sum() / n)
        mce = torch.maximum(mce, gap)
    return float(ece), float(mce)


@pytest.fixture
def binary_inputs() -> dict[str, object]:
    logits = torch.tensor(
        [[-2.0, 2.0], [2.0, -2.0], [-0.5, 0.5], [0.5, -0.5]],
        dtype=torch.float32,
    )
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)
    return {
        "clf": {"logits": logits, "probabilities": probs, "predictions": preds},
        "batch": {"y": targets},
    }


@pytest.fixture
def multiclass_inputs() -> dict[str, object]:
    logits = torch.tensor(
        [[4.0, 1.0, 0.0], [0.0, 4.0, 1.0], [0.0, 1.0, 4.0], [3.0, 2.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    return {"clf": {"logits": logits}, "batch": {"y": targets}}


@pytest.fixture
def regression_inputs() -> dict[str, object]:
    preds = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]], dtype=torch.float32)
    targets = torch.tensor([[1.5, 2.5], [2.5, 2.5], [2.5, 3.5]], dtype=torch.float32)
    return {"reg": {"predictions": preds}, "batch": {"target": targets}}


@pytest.fixture
def seg2d_inputs() -> dict[str, object]:
    logits = torch.tensor([[[[3.0, -3.0], [3.0, -3.0]]], [[[2.0, 2.0], [-2.0, -2.0]]]], dtype=torch.float32)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long().squeeze(1)
    targets = torch.tensor([[[1, 0], [1, 0]], [[1, 1], [0, 0]]], dtype=torch.long)
    return {"seg": {"logits": logits, "probabilities": probs, "predictions": preds}, "batch": {"mask": targets}}


@pytest.fixture
def seg3d_inputs() -> dict[str, object]:
    logits = torch.tensor([[[[[3.0, -3.0], [3.0, -3.0]], [[3.0, -3.0], [3.0, -3.0]]]]], dtype=torch.float32)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long().squeeze(1)
    targets = torch.tensor([[[[1, 0], [1, 0]], [[1, 0], [1, 0]]]], dtype=torch.long)
    return {"seg": {"logits": logits, "probabilities": probs, "predictions": preds}, "batch": {"mask": targets}}


def test_evaluator_resolve_and_composite_helpers() -> None:
    payload = {"x": torch.tensor([1.0, 3.0])}
    assert torch.equal(Evaluator.resolve(payload, "x"), payload["x"])

    report = CompositeReportEvaluator([DummyReportEvaluator("r1"), DummyReportEvaluator("r2")], name="report")
    metrics = report(inputs=payload)
    assert metrics["r1/mean"] == pytest.approx(2.0)
    assert metrics["r2/mean"] == pytest.approx(2.0)

    selector = CompositeSelectorEvaluator(
        [
            DummySelectorEvaluator(name="up", direction="maximize", weight=2.0),
            DummySelectorEvaluator(name="down", direction="minimize", weight=1.0),
        ],
        name="selector",
    )
    total, components = selector.compute(inputs=payload)
    assert float(total) == pytest.approx(2.0)
    assert components["selector/up"]["weighted"] == pytest.approx(4.0)
    assert components["selector/down"]["weighted"] == pytest.approx(-2.0)


def test_classification_report_and_selector_match(multiclass_inputs: dict[str, object]) -> None:
    report = ClassificationReportEvaluator(score_key="clf/logits", target_key="batch/y")
    selector = AccuracySelectorEvaluator(score_key="clf/logits", target_key="batch/y")

    metrics = report(inputs=multiclass_inputs)
    value = float(selector(inputs=multiclass_inputs))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert value == pytest.approx(metrics["accuracy"])


def test_calibration_report_and_selector_match_manual(binary_inputs: dict[str, object]) -> None:
    n_bins = 4
    report = CalibrationReportEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        n_bins=n_bins,
    )
    selector = BrierScoreSelectorEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
    )

    metrics = report(inputs=binary_inputs)
    selector_value = float(selector(inputs=binary_inputs))

    probs = torch.softmax(binary_inputs["clf"]["probabilities"], dim=1)[:, 1]  # type: ignore[index]
    targets = binary_inputs["batch"]["y"]  # type: ignore[index]
    expected_brier = float(torch.mean((probs - targets.float()) ** 2))
    expected_ece, expected_mce = _ece_and_mce(probs, targets, n_bins)

    assert metrics["brier"] == pytest.approx(expected_brier)
    assert metrics["ece"] == pytest.approx(expected_ece)
    assert metrics["mce"] == pytest.approx(expected_mce)
    assert selector_value == pytest.approx(expected_brier)


def test_regression_report_and_selector_match_manual(regression_inputs: dict[str, object]) -> None:
    report = RegressionReportEvaluator(pred_key="reg/predictions", target_key="batch/target")
    selector = MeanSquaredErrorSelectorEvaluator(pred_key="reg/predictions", target_key="batch/target")

    metrics = report(inputs=regression_inputs)
    selector_value = float(selector(inputs=regression_inputs))

    preds = regression_inputs["reg"]["predictions"]  # type: ignore[index]
    targets = regression_inputs["batch"]["target"]  # type: ignore[index]
    mse = ((preds - targets) ** 2).mean(dim=0)
    rmse = torch.sqrt(mse)

    assert metrics["mse"] == pytest.approx(float(mse.mean()))
    assert metrics["rmse"] == pytest.approx(float(rmse.mean()))
    assert metrics["mae"] == pytest.approx(float((preds - targets).abs().mean()))
    assert selector_value == pytest.approx(float(mse.mean()))


def test_roc_report_and_selector_match_manual(binary_inputs: dict[str, object]) -> None:
    report = ROCBinaryReportEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
    )
    selector = ROCAUCSelectorEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
    )

    metrics = report(inputs=binary_inputs)
    selector_value = float(selector(inputs=binary_inputs))

    probs = torch.softmax(binary_inputs["clf"]["probabilities"], dim=1)[:, 1]  # type: ignore[index]
    targets = binary_inputs["batch"]["y"]  # type: ignore[index]
    pos_scores = probs[targets == 1]
    neg_scores = probs[targets == 0]
    auc = sum(float(p > n) + 0.5 * float(p == n) for p in pos_scores for n in neg_scores) / (len(pos_scores) * len(neg_scores))

    assert metrics["auc"] == pytest.approx(auc)
    assert metrics["youden_j"] == pytest.approx(1.0)
    assert selector_value == pytest.approx(auc)


def test_binary_pr_auc_selector_matches_sklearn(binary_inputs: dict[str, object]) -> None:
    selector = BinaryPRAUCSelectorEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
    )

    selector_value = float(selector(inputs=binary_inputs))

    probs = torch.softmax(binary_inputs["clf"]["probabilities"], dim=1)[:, 1]  # type: ignore[index]
    targets = binary_inputs["batch"]["y"]  # type: ignore[index]
    expected = average_precision_score(
        targets.detach().cpu().numpy(),
        probs.detach().cpu().numpy(),
    )

    assert selector_value == pytest.approx(expected)


def test_roc_auc_selector_matches_sklearn(binary_inputs: dict[str, object]) -> None:
    selector = ROCAUCSelectorEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
    )

    selector_value = float(selector(inputs=binary_inputs))

    probs = torch.softmax(binary_inputs["clf"]["probabilities"], dim=1)[:, 1]  # type: ignore[index]
    targets = binary_inputs["batch"]["y"]  # type: ignore[index]
    expected = roc_auc_score(
        targets.detach().cpu().numpy(),
        probs.detach().cpu().numpy(),
    )

    assert selector_value == pytest.approx(expected)


def test_dca_report_and_selector_match_manual(binary_inputs: dict[str, object]) -> None:
    n_thresholds = 5
    report = DCAReportEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        n_thresholds=n_thresholds,
    )
    selector = MaximumNetBenefitSelectorEvaluator(
        score_key="clf/logits",
        target_key="batch/y",
        probabilities_key="clf/probabilities",
        n_thresholds=n_thresholds,
    )

    metrics = report(inputs=binary_inputs)
    selector_value = float(selector(inputs=binary_inputs))

    probs = torch.softmax(binary_inputs["clf"]["probabilities"], dim=1)[:, 1]  # type: ignore[index]
    targets = binary_inputs["batch"]["y"].float()  # type: ignore[index]
    thresholds = torch.linspace(0.01, 0.99, n_thresholds)
    curve = []
    for thr in thresholds:
        pred_pos = probs >= thr
        tp = ((pred_pos) & (targets == 1)).sum().float()
        fp = ((pred_pos) & (targets == 0)).sum().float()
        curve.append(float((tp / len(probs)) - (fp / len(probs)) * (thr / (1 - thr))))

    assert metrics["max_net_benefit"] == pytest.approx(max(curve))
    assert metrics["net_benefit_mean"] == pytest.approx(sum(curve) / len(curve))
    assert selector_value == pytest.approx(max(curve))


def test_segmentation_reports_and_selectors_match(seg2d_inputs: dict[str, object], seg3d_inputs: dict[str, object]) -> None:
    report_2d = SegmentationReportEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        probabilities_key="seg/probabilities",
    )
    selector_2d = SegmentationDiceSelectorEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        probabilities_key="seg/probabilities",
    )
    metrics_2d = report_2d(inputs=seg2d_inputs)
    selector_2d_value = float(selector_2d(inputs=seg2d_inputs))

    assert metrics_2d["dice"] == pytest.approx(1.0)
    assert metrics_2d["iou"] == pytest.approx(1.0)
    assert selector_2d_value == pytest.approx(metrics_2d["dice"])

    report_3d = Segmentation3DReportEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        probabilities_key="seg/probabilities",
        include_background=True,
    )
    selector_3d = Segmentation3DDiceSelectorEvaluator(
        score_key="seg/logits",
        target_key="batch/mask",
        probabilities_key="seg/probabilities",
        include_background=True,
    )
    try:
        metrics_3d = report_3d(inputs=seg3d_inputs)
    except RuntimeError as exc:
        if "Numpy is not available" in str(exc):
            pytest.skip("3D report evaluator requires Tensor.numpy() support in this environment.")
        raise
    selector_3d_value = float(selector_3d(inputs=seg3d_inputs))

    assert metrics_3d["dice"] == pytest.approx(1.0)
    assert metrics_3d["voxel_accuracy"] == pytest.approx(1.0)
    assert selector_3d_value == pytest.approx(metrics_3d["dice"])


def test_calibrated_logits_fallback_works_for_report_and_selector(binary_inputs: dict[str, object]) -> None:
    report = ClassificationReportEvaluator(score_key="clf/calibrated_logits", target_key="batch/y")
    selector = AccuracySelectorEvaluator(score_key="clf/calibrated_logits", target_key="batch/y")

    report_metrics = report(inputs=binary_inputs)
    selector_value = float(selector(inputs=binary_inputs))

    assert report_metrics["accuracy"] == pytest.approx(1.0)
    assert selector_value == pytest.approx(1.0)


def test_regression_report_rejects_shape_mismatch() -> None:
    report = RegressionReportEvaluator(pred_key="reg/predictions", target_key="batch/target")
    with pytest.raises(ValueError, match="identical shape"):
        report(
            inputs={
                "reg": {"predictions": torch.tensor([[1.0], [2.0]])},
                "batch": {"target": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
            }
        )


def test_selector_requires_scalar_tensor() -> None:
    class BadSelector(SelectorEvaluator):
        @property
        def required_keys(self) -> tuple[str, ...]:
            return ("x",)

        def primary_metric(self, *, inputs: dict[str, object]) -> torch.Tensor:
            return self.resolve(inputs, "x")

    selector = BadSelector(name="bad", direction="maximize")
    with pytest.raises(ValueError, match="scalar Tensor"):
        selector(inputs={"x": torch.tensor([1.0, 2.0])})


def test_report_values_are_finite(binary_inputs: dict[str, object], regression_inputs: dict[str, object]) -> None:
    classification = ClassificationReportEvaluator(score_key="clf/logits", target_key="batch/y")(inputs=binary_inputs)
    calibration = CalibrationReportEvaluator(score_key="clf/logits", target_key="batch/y")(inputs=binary_inputs)
    regression = RegressionReportEvaluator(pred_key="reg/predictions", target_key="batch/target")(inputs=regression_inputs)

    for metrics in (classification, calibration, regression):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert math.isfinite(value)

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch
from torch import Tensor

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from sktorch.modules.evaluation.classification import ClassificationEvaluator


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _np(x: Any) -> np.ndarray:
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _close(a: float, b: float, *, atol=1e-10, rtol=1e-10):
    if math.isnan(a) and math.isnan(b):
        return
    assert abs(a - b) <= atol + rtol * abs(b)


# ------------------------------------------------------------
# binary probabilistic
# ------------------------------------------------------------

def test_binary_probabilistic_matches_sklearn_and_is_stateful():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    p1 = np.array([0.1, 0.8, 0.55, 0.3, 0.9, 0.4, 0.2, 0.7], dtype=np.float32)
    y_prob = np.stack([1.0 - p1, p1], axis=1)
    y_pred = (p1 >= 0.5).astype(np.int64)

    ev = ClassificationEvaluator(
        pred_kind="probs",
        labels=(0, 1),
        selector="accuracy",
        return_pr_curve=True,
        return_normalized_confusion=True,
    )

    ev.reset()
    ev.update(
        predictions={"probs": torch.tensor(y_prob[:4])},
        targets={"y": torch.tensor(y_true[:4])},
    )
    ev.update(
        predictions={"probs": torch.tensor(y_prob[4:])},
        targets={"y": torch.tensor(y_true[4:])},
    )

    # state aggregation
    assert len(ev._y_true_parts) == 2
    assert len(ev._y_prob_parts) == 2

    metrics = ev.compute_metrics()

    # --- sklearn ground truth ---
    gt_acc = accuracy_score(y_true, y_pred)
    gt_bal = balanced_accuracy_score(y_true, y_pred)
    gt_ll = log_loss(y_true, y_prob, labels=[0, 1])
    gt_auc = roc_auc_score(y_true, y_prob[:, 1])
    gt_ap = average_precision_score(y_true, y_prob[:, 1])
    gt_brier = brier_score_loss(y_true, y_prob[:, 1])
    gt_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    gt_rep = classification_report(
        y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0
    )
    gt_p, gt_r, gt_thr = precision_recall_curve(y_true, y_prob[:, 1])

    # scalar metrics
    _close(metrics["accuracy"], gt_acc)
    _close(metrics["balanced_accuracy"], gt_bal)
    _close(metrics["log_loss"], gt_ll)
    _close(metrics["auc_binary"], gt_auc)
    _close(metrics["ap_binary"], gt_ap)
    _close(metrics["brier_score"], gt_brier)

    # macro stats
    _close(metrics["macro_f1"], gt_rep["macro avg"]["f1-score"])
    _close(metrics["weighted_f1"], gt_rep["weighted avg"]["f1-score"])
    _close(metrics["macro_precision"], gt_rep["macro avg"]["precision"])
    _close(metrics["macro_recall"], gt_rep["macro avg"]["recall"])

    # confusion matrix
    assert np.array_equal(_np(metrics["confusion_matrix"]), gt_cm)

    # PR curve
    assert np.allclose(_np(metrics["pr_curve/precision"]), gt_p)
    assert np.allclose(_np(metrics["pr_curve/recall"]), gt_r)
    assert np.allclose(_np(metrics["pr_curve/thresholds"]), gt_thr)


# ------------------------------------------------------------
# call() behaviour
# ------------------------------------------------------------

def test_call_updates_state_and_flattens_outputs():
    y_true = torch.tensor([0, 1, 0, 1])
    p1 = torch.tensor([0.2, 0.9, 0.4, 0.6])
    y_prob = torch.stack([1 - p1, p1], dim=1)

    ev = ClassificationEvaluator(
        pred_kind="probs",
        labels=(0, 1),
        selector="accuracy",
        return_pr_curve=False,
    )
    ev.reset()

    flat = ev(predictions={"probs": y_prob}, targets={"y": y_true})

    assert "cls" in flat
    assert "cls/accuracy" in flat
    assert "cls/confusion_matrix" in flat

    assert len(ev._y_true_parts) == 1


# ------------------------------------------------------------
# label contract enforcement
# ------------------------------------------------------------

def test_binary_probabilistic_requires_labels_0_1():
    y_true = torch.tensor([1, 2, 1, 2])
    p1 = torch.tensor([0.2, 0.8, 0.6, 0.4])
    y_prob = torch.stack([1 - p1, p1], dim=1)

    ev = ClassificationEvaluator(
        pred_kind="probs",
        labels=(1, 2),
        selector="accuracy",
    )
    ev.reset()
    ev.update(predictions={"probs": y_prob}, targets={"y": y_true})

    with pytest.raises(ValueError):
        ev.compute_metrics()


def test_multiclass_probabilistic_requires_zero_based_contiguous_labels():
    y_true = torch.tensor([1, 2, 3, 1, 2, 3])
    probs = torch.tensor(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )

    ev = ClassificationEvaluator(
        pred_kind="probs",
        labels=(1, 2, 3),
        selector="accuracy",
    )
    ev.reset()
    ev.update(predictions={"probs": probs}, targets={"y": y_true})

    with pytest.raises(ValueError):
        ev.compute_metrics()


# ------------------------------------------------------------
# multiclass logits
# ------------------------------------------------------------

def test_multiclass_logits_matches_sklearn():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    logits = np.array(
        [
            [2.0, 0.0, -1.0],
            [-1.0, 2.5, 0.1],
            [0.2, -0.2, 1.2],
            [1.5, 0.2, -0.5],
            [-0.3, 1.7, 0.0],
            [0.1, -0.1, 0.9],
        ],
        dtype=np.float32,
    )

    probs = _softmax(logits)
    y_pred = np.argmax(probs, axis=1)

    ev = ClassificationEvaluator(
        pred_kind="logits",
        labels=(0, 1, 2),
        selector="accuracy",
        return_pr_curve=False,
        return_normalized_confusion=False,
    )
    ev.reset()

    ev.update(
        predictions={"probs": torch.tensor(logits[:3])},
        targets={"y": torch.tensor(y_true[:3])},
    )
    ev.update(
        predictions={"probs": torch.tensor(logits[3:])},
        targets={"y": torch.tensor(y_true[3:])},
    )

    metrics = ev.compute_metrics()

    _close(metrics["accuracy"], accuracy_score(y_true, y_pred))
    _close(metrics["balanced_accuracy"], balanced_accuracy_score(y_true, y_pred))
    _close(metrics["log_loss"], log_loss(y_true, probs, labels=[0, 1, 2]))
    _close(metrics["auc_ovr_macro"], roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))

    assert np.array_equal(
        _np(metrics["confusion_matrix"]),
        confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
    )


# ------------------------------------------------------------
# misuse detection
# ------------------------------------------------------------

def test_inconsistent_probability_state_raises():
    y_true_a = torch.tensor([0, 1])
    y_true_b = torch.tensor([0, 1])
    p1 = torch.tensor([0.2, 0.8])
    y_prob = torch.stack([1 - p1, p1], dim=1)

    ev = ClassificationEvaluator(pred_kind="probs", labels=(0, 1), selector="accuracy")
    ev.reset()

    ev.update(predictions={"probs": y_prob}, targets={"y": y_true_a})

    # simulate misuse
    ev.pred_kind = "labels"
    ev.update(predictions={"probs": torch.tensor([0, 1])}, targets={"y": y_true_b})

    with pytest.raises(RuntimeError):
        ev.compute_metrics()

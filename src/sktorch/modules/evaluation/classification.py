from __future__ import annotations

from typing import Any, Dict, Mapping, Literal, Sequence

import numpy as np
from torch import Tensor

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from sktorch.modules.evaluation._base import RelationalEvaluator, SelectorSpec


def _to_numpy_1d(x: Any) -> np.ndarray:
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.reshape(-1)


def _to_numpy_2d(x: Any) -> np.ndarray:
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array/tensor for probabilities/logits, got shape {x.shape}.")
    return x


def _softmax_np(z: np.ndarray, axis: int = 1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _sigmoid_np(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _normalize_cm(cm: np.ndarray, *, mode: Literal["true", "pred", "all"]) -> np.ndarray:
    cm = cm.astype(np.float64, copy=False)
    if mode == "true":
        denom = cm.sum(axis=1, keepdims=True)
    elif mode == "pred":
        denom = cm.sum(axis=0, keepdims=True)
    else:  # "all"
        denom = cm.sum()
    denom = np.where(denom == 0, 1.0, denom)
    return cm / denom


def _is_contiguous_zero_based_labels(labels: np.ndarray, *, C: int) -> bool:
    """
    Option A contract:
    - For multiclass probabilistic metrics, we require labels == [0, 1, ..., C-1].
    """
    if labels.ndim != 1:
        return False
    if labels.size != C:
        return False
    # exact match in order (stronger than set equality, avoids silent column mismatch)
    return np.array_equal(labels, np.arange(C, dtype=labels.dtype))


def _raise_if_pred_not_subset_of_true(*, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    true_set = set(int(x) for x in np.unique(y_true).tolist())
    pred_set = set(int(x) for x in np.unique(y_pred).tolist())
    extra = sorted(pred_set - true_set)
    if extra:
        raise ValueError(
            f"{name}: invalid predictions: y_pred contains classes not present in y_true. "
            f"extra_pred_labels={extra}, true_labels={sorted(true_set)}"
        )


class ClassificationEvaluator(RelationalEvaluator):
    """
    Stateful classification evaluator producing discrete and probabilistic metrics.

    This evaluator aggregates predictions across multiple `update()` calls and
    computes metrics over the full accumulated dataset when `compute()` or
    `compute_metrics()` is invoked.

    --------------------------------------------------------------------------
    Expected Inputs
    --------------------------------------------------------------------------

    Predictions:
        predictions[pred_key] must be one of:

        - pred_kind="labels"
            Tensor of shape (N,) containing integer class predictions.

        - pred_kind="probs"
            Tensor of shape (N, C) containing class probabilities.
            Binary case may also be (N,) or (N,1) representing P(class=1).

        - pred_kind="logits"
            Tensor of shape (N, C) containing raw logits.
            Binary case may be (N,) or (N,1) logits for class 1.
            Logits are internally converted to probabilities.

    Targets:
        targets[target_key] must be integer class labels of shape (N,).

    --------------------------------------------------------------------------
    Strict Validation Rules
    --------------------------------------------------------------------------

    1) Prediction validity:
       All predicted labels must be a subset of the observed ground-truth labels.
       If y_pred contains classes not present in y_true, a ValueError is raised.

    2) Binary probabilistic metrics:
       When C == 2 and probabilities/logits are used:
       - Labels must be exactly {0, 1}.
       - Column 1 is treated as the positive class.

    3) Multiclass probabilistic metrics:
       When C > 2 and probabilities/logits are used:
       - Labels must be contiguous zero-based integers:
             [0, 1, ..., C-1]
       - The label order must exactly match probability column order.
       - Violations raise ValueError.

    4) Batch consistency:
       If some batches provide probabilities and others do not,
       a RuntimeError is raised during compute().

    --------------------------------------------------------------------------
    Metrics Produced
    --------------------------------------------------------------------------

    Always computed:
        - accuracy
        - balanced_accuracy
        - confusion_matrix
        - classification_report (sklearn format)
        - macro_precision / macro_recall / macro_f1
        - weighted_precision / weighted_recall / weighted_f1
        - mcc
        - kappa

    If probabilistic predictions are available:
        - log_loss
        - auc_binary (binary case)
        - auc_ovr_macro (multiclass case)
        - ap_binary (binary case)
        - ap_ovr_macro (multiclass case)
        - brier_score (binary only)
        - precision-recall curve(s)

    Additional normalized confusion matrices are optionally returned.

    --------------------------------------------------------------------------
    Stateful Lifecycle
    --------------------------------------------------------------------------

    Typical usage (epoch-level evaluation):

        evaluator.reset()

        for batch in dataloader:
            evaluator.update(predictions, targets)

        metrics = evaluator.compute_metrics()

    The metrics are computed over all accumulated batches.

    --------------------------------------------------------------------------
    __call__ Behavior (Inherited)
    --------------------------------------------------------------------------

    Calling the evaluator directly:

        flat_dict = evaluator(predictions=..., targets=...)

    will:
        - Validate inputs
        - Perform update()
        - Immediately compute aggregated metrics
        - Return a flattened dictionary of the form:

            {
                "{name}": selector_value,
                "{name}/accuracy": ...,
                "{name}/confusion_matrix": ...,
                ...
            }

    The selector value is determined by the `selector` specification
    provided at initialization.

    --------------------------------------------------------------------------
    Selector
    --------------------------------------------------------------------------

    The selector defines the scalar score used for model selection,
    early stopping, or pruning.

        selector = "accuracy"

    or

        selector = (("macro_f1", 0.5), ("auc_binary", 0.5))

    Weighted mixtures:
        - Ignore missing or NaN terms
        - Renormalize weights over valid terms
    """

    def __init__(
        self,
        *,
        name: str = "cls",
        pred_key: str = "probs",
        target_key: str = "y",
        pred_kind: Literal["probs", "logits", "labels"] = "probs",
        labels: Sequence[int] | None = None,
        zero_division: int = 0,
        selector: SelectorSpec = "accuracy",
        return_normalized_confusion: bool = True,
        return_pr_curve: bool = True,
        required: bool = True,
        required_context_keys: tuple[str, ...] | list[str] = (),
    ):
        self.pred_key = pred_key
        self.target_key = target_key
        self.pred_kind = pred_kind
        self.labels = tuple(labels) if labels is not None else None
        self.zero_division = int(zero_division)
        self.return_normalized_confusion = bool(return_normalized_confusion)
        self.return_pr_curve = bool(return_pr_curve)

        self._y_true_parts: list[np.ndarray] = []
        self._y_pred_parts: list[np.ndarray] = []
        self._y_prob_parts: list[np.ndarray] = []
        self._saw_probabilities: bool = False

        super().__init__(
            name=name,
            selector=selector,
            required=required,
            required_pred_keys=(pred_key,),
            required_target_keys=(target_key,),
            required_context_keys=required_context_keys,
        )

    def reset(self) -> None:
        self._y_true_parts.clear()
        self._y_pred_parts.clear()
        self._y_prob_parts.clear()
        self._saw_probabilities = False

    def update(
        self,
        predictions: Mapping[str, Tensor | None],
        targets: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> None:
        y_true = _to_numpy_1d(targets[self.target_key])

        y_prob: np.ndarray | None = None
        if self.pred_kind == "labels":
            y_pred = _to_numpy_1d(predictions[self.pred_key])
        else:
            raw = _to_numpy_2d(predictions[self.pred_key])

            # Binary special-cases: raw may be (N,1) representing score/prob for class 1
            if raw.shape[1] == 1:
                score1 = raw.reshape(-1)
                p1 = _sigmoid_np(score1) if self.pred_kind == "logits" else score1
                y_prob = np.stack([1.0 - p1, p1], axis=1)
            else:
                y_prob = _softmax_np(raw, axis=1) if self.pred_kind == "logits" else raw

            y_pred = np.argmax(y_prob, axis=1)

        self._y_true_parts.append(y_true.astype(np.int64, copy=False))
        self._y_pred_parts.append(np.asarray(y_pred).reshape(-1).astype(np.int64, copy=False))

        if y_prob is not None:
            self._y_prob_parts.append(np.asarray(y_prob, dtype=np.float32))
            self._saw_probabilities = True

    def compute_metrics(self) -> Dict[str, Any]:
        if not self._y_true_parts:
            return {"empty": True}

        y_true = np.concatenate(self._y_true_parts, axis=0)
        y_pred = np.concatenate(self._y_pred_parts, axis=0)

        _raise_if_pred_not_subset_of_true(y_true=y_true, y_pred=y_pred, name=self.__class__.__name__)

        # label ordering for discrete metrics (CM/report): either user-provided or inferred from observed labels
        if self.labels is not None:
            labels = np.asarray(self.labels, dtype=np.int64)
        else:
            labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)])).astype(np.int64, copy=False)

        # core scalars
        acc = float(accuracy_score(y_true, y_pred))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        mcc = float(matthews_corrcoef(y_true, y_pred)) if labels.size >= 2 else float("nan")
        kappa = float(cohen_kappa_score(y_true, y_pred)) if labels.size >= 2 else float("nan")

        rep = classification_report(
            y_true,
            y_pred,
            labels=labels.tolist(),
            output_dict=True,
            zero_division=self.zero_division,
        )

        cm = confusion_matrix(y_true, y_pred, labels=labels.tolist())

        metrics: Dict[str, Any] = {
            "labels": labels,
            "confusion_matrix": cm,
            "classification_report": rep,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "mcc": mcc,
            "kappa": kappa,
        }

        if self.return_normalized_confusion:
            metrics["confusion_matrix_norm_true"] = _normalize_cm(cm, mode="true")
            metrics["confusion_matrix_norm_pred"] = _normalize_cm(cm, mode="pred")
            metrics["confusion_matrix_norm_all"] = _normalize_cm(cm, mode="all")

        for avg_key, prefix in (("macro avg", "macro"), ("weighted avg", "weighted")):
            if avg_key in rep:
                metrics[f"{prefix}_precision"] = float(rep[avg_key]["precision"])
                metrics[f"{prefix}_recall"] = float(rep[avg_key]["recall"])
                metrics[f"{prefix}_f1"] = float(rep[avg_key]["f1-score"])

        # probabilistic metrics
        auc_binary = float("nan")
        auc_ovr_macro = float("nan")
        ap_binary = float("nan")
        ap_ovr_macro = float("nan")
        ll = float("nan")
        brier = float("nan")

        y_prob: np.ndarray | None = None
        if self._saw_probabilities:
            if len(self._y_prob_parts) != len(self._y_true_parts):
                raise RuntimeError(
                    f"{self.__class__.__name__}: inconsistent state: saw probabilities for some batches but not all. "
                    f"Ensure pred_kind is consistent and predictions always include '{self.pred_key}'."
                )
            y_prob = np.concatenate(self._y_prob_parts, axis=0)

        if y_prob is not None:
            C = int(y_prob.shape[1])

            # --- Option A enforcement (prevents silent column/label mismatch) ---
            if C == 2:
                # require binary labels {0,1}
                # (order can be [0,1] or [1,0] for discrete metrics, but probabilistic metrics assume class 1 is positive)
                # We enforce that label set is exactly {0,1}.
                label_set = set(int(x) for x in np.unique(labels).tolist())
                if label_set != {0, 1}:
                    raise ValueError(
                        f"{self.__class__.__name__}: binary probabilistic metrics require labels {{0,1}}, "
                        f"got labels={labels.tolist()}."
                    )
            elif C > 2:
                # require labels == [0..C-1] exactly
                if not _is_contiguous_zero_based_labels(labels, C=C):
                    raise ValueError(
                        f"{self.__class__.__name__}: multiclass probabilistic metrics require labels == "
                        f"[0, 1, ..., C-1] with C={C} matching probability columns. "
                        f"Got labels={labels.tolist()}."
                    )

            # log-loss
            try:
                # For Option A: labels passed here must align with columns; our enforcement ensures that.
                ll = float(log_loss(y_true, y_prob, labels=labels.tolist()))
            except Exception:
                ll = float("nan")

            # ROC-AUC
            try:
                if C == 2:
                    auc_binary = float(roc_auc_score(y_true, y_prob[:, 1]))
                elif C > 2:
                    auc_ovr_macro = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
            except Exception:
                pass

            # Average Precision / PR-AUC
            try:
                if C == 2:
                    ap_binary = float(average_precision_score(y_true, y_prob[:, 1]))
                elif C > 2:
                    y_bin = label_binarize(y_true, classes=labels.tolist())
                    ap_ovr_macro = float(average_precision_score(y_bin, y_prob, average="macro"))
            except Exception:
                pass

            # Brier score (binary only)
            try:
                if C == 2:
                    brier = float(brier_score_loss(y_true, y_prob[:, 1]))
            except Exception:
                pass

            # Precision-Recall curve(s)
            if self.return_pr_curve:
                try:
                    if C == 2:
                        p, r, thr = precision_recall_curve(y_true, y_prob[:, 1])
                        metrics["pr_curve/precision"] = p
                        metrics["pr_curve/recall"] = r
                        metrics["pr_curve/thresholds"] = thr
                    elif C > 2:
                        y_bin = label_binarize(y_true, classes=labels.tolist())
                        per_class: Dict[str, Any] = {}
                        for j, lab in enumerate(labels.tolist()):
                            p, r, thr = precision_recall_curve(y_bin[:, j], y_prob[:, j])
                            per_class[str(lab)] = {"precision": p, "recall": r, "thresholds": thr}
                        metrics["pr_curve/per_class"] = per_class
                except Exception:
                    pass

        metrics["log_loss"] = ll
        metrics["brier_score"] = brier
        metrics["auc_binary"] = auc_binary
        metrics["auc_ovr_macro"] = auc_ovr_macro
        metrics["ap_binary"] = ap_binary
        metrics["ap_ovr_macro"] = ap_ovr_macro

        return metrics

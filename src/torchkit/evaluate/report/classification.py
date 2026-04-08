from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from torchkit.evaluate.report._report_evaluator import ReportEvaluator
from torchkit.evaluate.mixins._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class ClassificationReportEvaluator(CalibratedLogitsFallbackMixin, ReportEvaluator):
    """
    Evaluates classification outputs.

    Expected tensors:
        - score tensor: (N,), (N,1), or (N,C), typically logits or calibrated_logits
        - targets:      (N,)
        - optionally probabilities: (N,), (N,1), or (N,C)
        - optionally predictions:   (N,)
    """

    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "classification",
    ) -> None:
        super().__init__(name=name)

        self.score_key = score_key
        self.target_key = target_key
        self.probabilities_key = probabilities_key
        self.predictions_key = predictions_key

    @property
    def required_keys(self) -> tuple[str, ...]:
        keys = [self.score_key, self.target_key]
        if self.probabilities_key is not None:
            keys.append(self.probabilities_key)
        if self.predictions_key is not None:
            keys.append(self.predictions_key)
        return tuple(keys)

    @staticmethod
    def _scores_to_probabilities(scores: Tensor) -> Tensor:
        if scores.ndim == 1:
            p1 = torch.sigmoid(scores)
            return torch.stack([1.0 - p1, p1], dim=1)
        if scores.ndim == 2 and scores.shape[1] == 1:
            p1 = torch.sigmoid(scores[:, 0])
            return torch.stack([1.0 - p1, p1], dim=1)
        if scores.ndim == 2:
            return torch.softmax(scores, dim=1)
        raise ValueError(
            f"score tensor must be shape (N,), (N,1), or (N,C), got {tuple(scores.shape)}"
        )

    @staticmethod
    def _normalize_probabilities(probs: Tensor) -> Tensor:
        if probs.ndim == 1:
            return torch.stack([1.0 - probs, probs], dim=1)
        if probs.ndim == 2 and probs.shape[1] == 1:
            p1 = probs[:, 0]
            return torch.stack([1.0 - p1, p1], dim=1)
        if probs.ndim == 2:
            return probs
        raise ValueError(
            f"probabilities must be shape (N,), (N,1), or (N,C), got {tuple(probs.shape)}"
        )

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        scores = self._resolve_score_tensor(inputs)
        targets = self.resolve(inputs, self.target_key).detach()

        if targets.ndim != 1:
            raise ValueError(f"targets must be shape (N,), got {tuple(targets.shape)}")

        if self.probabilities_key is not None:
            probs_raw = self.resolve(inputs, self.probabilities_key).detach()
            probs = self._normalize_probabilities(probs_raw)
        else:
            probs = self._scores_to_probabilities(scores)

        if probs.ndim != 2:
            raise ValueError(f"probabilities must resolve to shape (N,C), got {tuple(probs.shape)}")

        n, c = probs.shape

        if targets.shape[0] != n:
            raise ValueError(
                f"targets batch size {targets.shape[0]} does not match probability batch size {n}"
            )

        if self.predictions_key is not None:
            preds = self.resolve(inputs, self.predictions_key).detach()
            if preds.ndim != 1:
                raise ValueError(f"predictions must be shape (N,), got {tuple(preds.shape)}")
            if preds.shape[0] != n:
                raise ValueError(
                    f"predictions batch size {preds.shape[0]} does not match probability batch size {n}"
                )
            preds = preds.long()
        else:
            preds = torch.argmax(probs, dim=1)

        device = probs.device
        dtype = probs.dtype

        cm = torch.zeros(c, c, device=device, dtype=torch.long)
        for t, p in zip(targets, preds):
            cm[t, p] += 1

        tp = cm.diag().to(dtype)
        fp = cm.sum(0).to(dtype) - tp
        fn = cm.sum(1).to(dtype) - tp
        support = cm.sum(1).to(dtype)

        eps = torch.tensor(1e-12, device=device, dtype=dtype)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()

        micro_tp = tp.sum()
        micro_fp = fp.sum()
        micro_fn = fn.sum()

        micro_precision = micro_tp / (micro_tp + micro_fp + eps)
        micro_recall = micro_tp / (micro_tp + micro_fn + eps)
        micro_f1 = 2 * micro_precision * micro_recall / (
            micro_precision + micro_recall + eps
        )

        accuracy = (preds == targets).float().mean()
        balanced_accuracy = recall.mean()

        if c == 2:
            from sklearn.metrics import average_precision_score, precision_recall_curve

            pos_probs = probs[:, 1]
            targets_np = targets.detach().cpu().numpy()
            pos_probs_np = pos_probs.detach().cpu().numpy()
            pr_curve_precision_np, pr_curve_recall_np, thresholds_np = precision_recall_curve(
                targets_np,
                pos_probs_np,
            )
            pr_curve_precision = torch.as_tensor(pr_curve_precision_np.copy(), device=device, dtype=dtype)
            pr_curve_recall = torch.as_tensor(pr_curve_recall_np.copy(), device=device, dtype=dtype)
            thresholds = torch.as_tensor(thresholds_np.copy(), device=device, dtype=dtype)
            pr_auc = torch.tensor(
                float(average_precision_score(targets_np, pos_probs_np)),
                device=device,
                dtype=dtype,
            )
        else:
            pr_auc = torch.tensor(0.0, device=device, dtype=dtype)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.cpu().tolist(),
        }

        for cls_idx in range(c):
            metrics[f"precision/class_{cls_idx}"] = float(precision[cls_idx])
            metrics[f"recall/class_{cls_idx}"] = float(recall[cls_idx])
            metrics[f"f1/class_{cls_idx}"] = float(f1[cls_idx])
            metrics[f"support/class_{cls_idx}"] = int(support[cls_idx])

        if c == 2:
            metrics["pr_curve"] = {
                "precision": pr_curve_precision.cpu().tolist(),
                "recall": pr_curve_recall.cpu().tolist(),
                "thresholds": thresholds.cpu().tolist(),
            }

        return metrics

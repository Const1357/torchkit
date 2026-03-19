from __future__ import annotations

from typing import Any, Optional

import torch

from torchkit.evaluate.report._report_evaluator import ReportEvaluator
from torchkit.evaluate.mixins._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class ROCBinaryReportEvaluator(CalibratedLogitsFallbackMixin, ReportEvaluator):
    """
    Binary ROC evaluation.

    Computes:
        - ROC curve
        - ROC AUC
        - Youden's J statistic
        - Optimal threshold (max J)

    Expected inputs:
        - score tensor: (N,2), (N,1), or (N,)
        - targets:      (N,) with {0,1}
        - optionally probabilities: (N,2), (N,1), or (N,)
    """

    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "roc",
    ) -> None:
        super().__init__(name=name)

        self.score_key = score_key
        self.target_key = target_key
        self.probabilities_key = probabilities_key

    @property
    def required_keys(self) -> tuple[str, ...]:
        keys = [self.score_key, self.target_key]
        if self.probabilities_key is not None:
            keys.append(self.probabilities_key)
        return tuple(keys)

    @staticmethod
    def _binary_positive_probability(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2 and x.shape[1] == 2:
            return torch.softmax(x, dim=1)[:, 1]
        if x.ndim == 2 and x.shape[1] == 1:
            return torch.sigmoid(x[:, 0])
        if x.ndim == 1:
            return torch.sigmoid(x)
        raise ValueError(
            f"Expected binary scores/probabilities of shape (N,), (N,1), or (N,2), got {tuple(x.shape)}."
        )

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        scores_tensor = self._resolve_score_tensor(inputs)
        targets = self.resolve(inputs, self.target_key).detach()

        if targets.ndim != 1:
            raise ValueError(f"targets must be shape (N,), got {tuple(targets.shape)}")

        if self.probabilities_key is not None:
            probs_tensor = self.resolve(inputs, self.probabilities_key).detach()
            scores = self._binary_positive_probability(probs_tensor)
        else:
            scores = self._binary_positive_probability(scores_tensor)

        if scores.ndim != 1:
            raise ValueError(f"Resolved positive-class scores must be shape (N,), got {tuple(scores.shape)}")

        if scores.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Scores batch size {scores.shape[0]} does not match targets batch size {targets.shape[0]}"
            )

        device = scores.device
        dtype = scores.dtype

        order = torch.argsort(scores, descending=True)
        scores = scores[order]
        targets = targets[order]

        pos = targets == 1
        neg = targets == 0

        p = pos.sum().to(dtype)
        n = neg.sum().to(dtype)

        if p.item() == 0 or n.item() == 0:
            raise ValueError(
                f"ROCBinaryReportEvaluator requires both positive and negative samples. Got positives={int(p.item())}, negatives={int(n.item())}."
            )

        tp = torch.cumsum(pos.to(dtype), dim=0)
        fp = torch.cumsum(neg.to(dtype), dim=0)

        tpr = tp / p
        fpr = fp / n

        zero = torch.zeros(1, device=device, dtype=dtype)
        tpr = torch.cat([zero, tpr])
        fpr = torch.cat([zero, fpr])
        thresholds = torch.cat([torch.tensor([1.0], device=device, dtype=dtype), scores])

        auc = torch.trapz(tpr, fpr)

        j_stat = tpr - fpr
        j_idx = torch.argmax(j_stat)

        best_threshold = thresholds[j_idx]
        best_tpr = tpr[j_idx]
        best_fpr = fpr[j_idx]

        specificity = 1.0 - best_fpr
        sensitivity = best_tpr

        return {
            "auc": float(auc),
            "youden_j": float(j_stat[j_idx]),
            "youden_threshold": float(best_threshold),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "tpr": float(best_tpr),
            "fpr": float(best_fpr),
            "roc_curve": {
                "tpr": tpr.cpu().tolist(),
                "fpr": fpr.cpu().tolist(),
                "thresholds": thresholds.cpu().tolist(),
            },
            "roc_curve/tpr_mean": float(tpr.mean()),
            "roc_curve/fpr_mean": float(fpr.mean()),
        }

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from torchkit.evaluate._evaluator import Evaluator, MetricDirection


class ROCBinaryEvaluator(Evaluator):
    """
    Binary ROC evaluation.

    Computes:
        - ROC curve
        - ROC AUC
        - Youden's J statistic
        - Optimal threshold (max J)

    Expected inputs:
        logits:  (N,2) or (N,)
        targets: (N,) with {0,1}
    """

    def __init__(
        self,
        *,
        logits_key: str,
        target_key: str,
        name: str = "roc",
        primary_metric: str = "auc",
        direction: MetricDirection = "maximize",
        weight: float = 1.0,
    ):
        super().__init__(
            name=name,
            primary_metric=primary_metric,
            direction=direction,
            weight=weight,
        )

        self.logits_key = logits_key
        self.target_key = target_key

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.logits_key, self.target_key)

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:

        logits = self.resolve(inputs, self.logits_key).detach()
        targets = self.resolve(inputs, self.target_key).detach()

        if targets.ndim != 1:
            raise ValueError("targets must be shape (N,)")

        if logits.ndim == 2:
            scores = torch.softmax(logits, dim=1)[:, 1]
        elif logits.ndim == 1:
            scores = torch.sigmoid(logits)
        else:
            raise ValueError("logits must be shape (N,) or (N,2)")

        device = scores.device
        dtype = scores.dtype

        # sort by score descending
        order = torch.argsort(scores, descending=True)

        scores = scores[order]
        targets = targets[order]

        pos = targets == 1
        neg = targets == 0

        P = pos.sum().to(dtype)
        N = neg.sum().to(dtype)

        # cumulative TP / FP
        tp = torch.cumsum(pos.to(dtype), dim=0)
        fp = torch.cumsum(neg.to(dtype), dim=0)

        tpr = tp / P
        fpr = fp / N

        # prepend (0,0)
        zero = torch.zeros(1, device=device, dtype=dtype)

        tpr = torch.cat([zero, tpr])
        fpr = torch.cat([zero, fpr])
        thresholds = torch.cat(
            [torch.tensor([1.0], device=device, dtype=dtype), scores]
        )

        # exact ROC AUC
        auc = torch.trapz(tpr, fpr)

        # Youden's J
        J = tpr - fpr

        j_idx = torch.argmax(J)

        best_threshold = thresholds[j_idx]
        best_tpr = tpr[j_idx]
        best_fpr = fpr[j_idx]

        specificity = 1.0 - best_fpr
        sensitivity = best_tpr

        return {

            "auc": float(auc),

            "youden_j": float(J[j_idx]),
            "youden_threshold": float(best_threshold),

            "sensitivity": float(sensitivity),
            "specificity": float(specificity),

            "tpr": float(best_tpr),
            "fpr": float(best_fpr),

            # full ROC curve
            "roc_curve": {
                "tpr": tpr.cpu().tolist(),
                "fpr": fpr.cpu().tolist(),
                "thresholds": thresholds.cpu().tolist(),
            },

            # summaries
            "roc_curve/tpr_mean": float(tpr.mean()),
            "roc_curve/fpr_mean": float(fpr.mean()),
        }
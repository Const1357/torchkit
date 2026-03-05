from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from torchkit.evaluate._evaluator import Evaluator, MetricDirection

class ClassificationEvaluator(Evaluator):
    """
    Evaluates classification outputs.

    Expected tensors:
        logits:  (N, C)
        targets: (N,)
    """

    def __init__(
        self,
        *,
        logits_key: str,
        target_key: str,
        name: str = "classification",
        primary_metric: str = "macro_f1",
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

    #  required keys ---

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.logits_key, self.target_key)

    # metric computation ---

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:

        logits = self.resolve(inputs, self.logits_key).detach()
        targets = self.resolve(inputs, self.target_key).detach()

        if logits.ndim != 2:
            raise ValueError(f"logits must be shape (N,C), got {tuple(logits.shape)}")

        if targets.ndim != 1:
            raise ValueError(f"targets must be shape (N,), got {tuple(targets.shape)}")

        N, C = logits.shape

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        device = logits.device
        dtype = logits.dtype

        # confusion matrix ---

        cm = torch.zeros(C, C, device=device, dtype=torch.long)

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

        # aggregates ---

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

        # PR curve ---

        pr_points = 100
        thresholds = torch.linspace(0, 1, pr_points, device=device)

        pr_curve_precision = torch.zeros(pr_points, device=device, dtype=dtype)
        pr_curve_recall = torch.zeros(pr_points, device=device, dtype=dtype)

        if C == 2:
            pos_probs = probs[:, 1]

            for i, thr in enumerate(thresholds):

                pred_pos = pos_probs >= thr

                tp_t = ((pred_pos) & (targets == 1)).sum().to(dtype)
                fp_t = ((pred_pos) & (targets == 0)).sum().to(dtype)
                fn_t = ((~pred_pos) & (targets == 1)).sum().to(dtype)

                pr_curve_precision[i] = tp_t / (tp_t + fp_t + eps)
                pr_curve_recall[i] = tp_t / (tp_t + fn_t + eps)

            # AUC-PR (trapezoidal rule)
            pr_auc = torch.trapz(pr_curve_precision, pr_curve_recall)

        else:
            pr_auc = torch.tensor(0.0, device=device, dtype=dtype)

        # metrics dict ---

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
        }

        # per-class metrics
        for c in range(C):
            metrics[f"precision/class_{c}"] = float(precision[c])
            metrics[f"recall/class_{c}"] = float(recall[c])
            metrics[f"f1/class_{c}"] = float(f1[c])
            metrics[f"support/class_{c}"] = int(support[c])

        # confusion matrix (as nested list)
        metrics["confusion_matrix"] = cm.cpu().tolist()

        # PR curve
        if C == 2:
            metrics["pr_curve"] = {
                "precision": pr_curve_precision.cpu().tolist(),
                "recall": pr_curve_recall.cpu().tolist(),
                "thresholds": thresholds.cpu().tolist(),
            }

        return metrics
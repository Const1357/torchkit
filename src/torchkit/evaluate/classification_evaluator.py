from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from torchkit.evaluate._evaluator import Evaluator, MetricDirection


class ClassificationEvaluator(Evaluator):
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

    def _resolve_score_tensor(self, inputs: dict[str, Any]) -> Tensor:
        try:
            return self.resolve(inputs, self.score_key).detach()
        except KeyError:
            if self.score_key.endswith("calibrated_logits"):
                fallback_key = self.score_key[: -len("calibrated_logits")] + "logits"
                return self.resolve(inputs, fallback_key).detach()
            raise

    @staticmethod
    def _infer_num_classes(x: Tensor) -> int:
        if x.ndim == 1:
            return 2
        if x.ndim == 2 and x.shape[1] == 1:
            return 2
        if x.ndim == 2:
            return int(x.shape[1])
        raise ValueError(f"score/probability tensor must be shape (N,), (N,1), or (N,C), got {tuple(x.shape)}")

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
        raise ValueError(f"score tensor must be shape (N,), (N,1), or (N,C), got {tuple(scores.shape)}")

    @staticmethod
    def _normalize_probabilities(probs: Tensor) -> Tensor:
        if probs.ndim == 1:
            return torch.stack([1.0 - probs, probs], dim=1)
        if probs.ndim == 2 and probs.shape[1] == 1:
            p1 = probs[:, 0]
            return torch.stack([1.0 - p1, p1], dim=1)
        if probs.ndim == 2:
            return probs
        raise ValueError(f"probabilities must be shape (N,), (N,1), or (N,C), got {tuple(probs.shape)}")

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

        N, C = probs.shape

        if targets.shape[0] != N:
            raise ValueError(
                f"targets batch size {targets.shape[0]} does not match probability batch size {N}"
            )

        if self.predictions_key is not None:
            preds = self.resolve(inputs, self.predictions_key).detach()
            if preds.ndim != 1:
                raise ValueError(f"predictions must be shape (N,), got {tuple(preds.shape)}")
            if preds.shape[0] != N:
                raise ValueError(
                    f"predictions batch size {preds.shape[0]} does not match probability batch size {N}"
                )
            preds = preds.long()
        else:
            preds = torch.argmax(probs, dim=1)

        device = probs.device
        dtype = probs.dtype

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

            pr_auc = torch.trapz(pr_curve_precision, pr_curve_recall)
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
        }

        for c in range(C):
            metrics[f"precision/class_{c}"] = float(precision[c])
            metrics[f"recall/class_{c}"] = float(recall[c])
            metrics[f"f1/class_{c}"] = float(f1[c])
            metrics[f"support/class_{c}"] = int(support[c])

        metrics["confusion_matrix"] = cm.cpu().tolist()

        if C == 2:
            metrics["pr_curve"] = {
                "precision": pr_curve_precision.cpu().tolist(),
                "recall": pr_curve_recall.cpu().tolist(),
                "thresholds": thresholds.cpu().tolist(),
            }

        return metrics
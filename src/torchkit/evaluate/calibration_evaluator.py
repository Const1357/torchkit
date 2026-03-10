from __future__ import annotations

from typing import Any, Optional

import torch

from torchkit.evaluate._evaluator import Evaluator, MetricDirection
from torchkit.evaluate._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class CalibrationEvaluator(CalibratedLogitsFallbackMixin, Evaluator):
    """
    Probability calibration evaluation.

    Computes:
        - Brier score
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        - Calibration curve

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
        name: str = "calibration",
        primary_metric: str = "brier",
        direction: MetricDirection = "minimize",
        weight: float = 1.0,
        n_bins: int = 15,
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
        self.n_bins = n_bins

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
        raise ValueError(f"Expected binary scores/probabilities of shape (N,), (N,1), or (N,2), got {tuple(x.shape)}.")

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:

        scores = self._resolve_score_tensor(inputs)
        targets = self.resolve(inputs, self.target_key).detach()

        if targets.ndim != 1:
            raise ValueError(f"targets must be shape (N,), got {tuple(targets.shape)}")

        if self.probabilities_key is not None:
            probs_tensor = self.resolve(inputs, self.probabilities_key).detach()
            probs = self._binary_positive_probability(probs_tensor)
        else:
            probs = self._binary_positive_probability(scores)

        if probs.ndim != 1:
            raise ValueError(f"Resolved positive-class probabilities must be shape (N,), got {tuple(probs.shape)}")

        if probs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Probabilities batch size {probs.shape[0]} does not match targets batch size {targets.shape[0]}"
            )

        targets = targets.float()

        brier = torch.mean((probs - targets) ** 2)

        bin_edges = torch.linspace(0, 1, self.n_bins + 1, device=probs.device)

        bin_confidence = []
        bin_accuracy = []

        ece = torch.tensor(0.0, device=probs.device)
        mce = torch.tensor(0.0, device=probs.device)

        N = len(probs)

        for i in range(self.n_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]

            if i == self.n_bins - 1:
                mask = (probs >= lo) & (probs <= hi)
            else:
                mask = (probs >= lo) & (probs < hi)

            if mask.sum() == 0:
                continue

            p_bin = probs[mask]
            y_bin = targets[mask]

            conf = p_bin.mean()
            acc = y_bin.mean()

            gap = torch.abs(acc - conf)

            weight = mask.sum() / N

            ece += gap * weight
            mce = torch.maximum(mce, gap)

            bin_confidence.append(conf)
            bin_accuracy.append(acc)

        if len(bin_confidence) == 0:
            bin_confidence_t = torch.tensor([], device=probs.device)
            bin_accuracy_t = torch.tensor([], device=probs.device)
        else:
            bin_confidence_t = torch.stack(bin_confidence)
            bin_accuracy_t = torch.stack(bin_accuracy)

        return {
            "brier": float(brier),
            "ece": float(ece),
            "mce": float(mce),
            "calibration_curve": {
                "confidence": bin_confidence_t.cpu().tolist(),
                "accuracy": bin_accuracy_t.cpu().tolist(),
                "bin_edges": bin_edges.cpu().tolist(),
            },
        }
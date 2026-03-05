from __future__ import annotations

from typing import Any

import torch

from torchkit.evaluate._evaluator import Evaluator, MetricDirection


class CalibrationEvaluator(Evaluator):
    """
    Probability calibration evaluation.

    Computes:
        - Brier score
        - Expected Calibration Error (ECE)
        - Maximum Calibration Error (MCE)
        - Calibration curve

    Expected inputs:
        logits:  (N,2) or (N,)
        targets: (N,) with {0,1}
    """

    def __init__(
        self,
        *,
        logits_key: str,
        target_key: str,
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

        self.logits_key = logits_key
        self.target_key = target_key
        self.n_bins = n_bins

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.logits_key, self.target_key)

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:

        logits = self.resolve(inputs, self.logits_key).detach()
        targets = self.resolve(inputs, self.target_key).detach()

        if logits.ndim == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]
        elif logits.ndim == 1:
            probs = torch.sigmoid(logits)
        else:
            raise ValueError("logits must be shape (N,) or (N,2)")

        targets = targets.float()

        # ---------------- Brier score ----------------

        brier = torch.mean((probs - targets) ** 2)

        # ---------------- Calibration bins ----------------

        bin_edges = torch.linspace(0, 1, self.n_bins + 1, device=probs.device)

        bin_confidence = []
        bin_accuracy = []
        bin_count = []

        ece = torch.tensor(0.0, device=probs.device)
        mce = torch.tensor(0.0, device=probs.device)

        N = len(probs)

        for i in range(self.n_bins):

            lo = bin_edges[i]
            hi = bin_edges[i + 1]

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
            bin_count.append(mask.sum())

        if len(bin_confidence) == 0:
            # degenerate case
            bin_confidence = torch.tensor([], device=probs.device)
            bin_accuracy = torch.tensor([], device=probs.device)
        else:
            bin_confidence = torch.stack(bin_confidence)
            bin_accuracy = torch.stack(bin_accuracy)

        return {

            "brier": float(brier),

            "ece": float(ece),
            "mce": float(mce),

            "calibration_curve": {
                "confidence": bin_confidence.cpu().tolist(),
                "accuracy": bin_accuracy.cpu().tolist(),
                "bin_edges": bin_edges.cpu().tolist(),
            }
        }
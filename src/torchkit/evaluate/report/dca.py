from __future__ import annotations

from typing import Any, Optional

import torch

from torchkit.evaluate.report._report_evaluator import ReportEvaluator
from torchkit.evaluate.mixins._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class DCAReportEvaluator(CalibratedLogitsFallbackMixin, ReportEvaluator):
    """
    Decision Curve Analysis (DCA).

    Computes:
        - Net benefit curve
        - Treat-all net benefit
        - Treat-none baseline
        - Best threshold by net benefit

    Expected inputs:
        - score tensor: (N,), (N,1), or (N,2)
        - targets:      (N,) with {0,1}
        - optionally probabilities: (N,), (N,1), or (N,2)
    """

    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "dca",
        n_thresholds: int = 100,
    ) -> None:
        super().__init__(name=name)

        self.score_key = score_key
        self.target_key = target_key
        self.probabilities_key = probabilities_key
        self.n_thresholds = int(n_thresholds)

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
            raise ValueError(
                f"Resolved positive-class probabilities must be shape (N,), got {tuple(probs.shape)}"
            )

        if probs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Probabilities batch size {probs.shape[0]} does not match targets batch size {targets.shape[0]}"
            )

        targets = targets.float()
        n = len(probs)
        thresholds = torch.linspace(0.01, 0.99, self.n_thresholds, device=probs.device)

        net_benefit = []
        treat_all = []
        prevalence = targets.mean()

        for t in thresholds:
            pred_pos = probs >= t
            tp = ((pred_pos) & (targets == 1)).sum().float()
            fp = ((pred_pos) & (targets == 0)).sum().float()

            weight = t / (1 - t)
            nb = (tp / n) - (fp / n) * weight
            net_benefit.append(nb)

            nb_all = prevalence - (1 - prevalence) * weight
            treat_all.append(nb_all)

        net_benefit = torch.stack(net_benefit)
        treat_all = torch.stack(treat_all)
        treat_none = torch.zeros_like(net_benefit)

        best_idx = torch.argmax(net_benefit)
        best_threshold = thresholds[best_idx]
        best_nb = net_benefit[best_idx]

        return {
            "max_net_benefit": float(best_nb),
            "best_threshold": float(best_threshold),
            "net_benefit_mean": float(net_benefit.mean()),
            "dca_curve": {
                "thresholds": thresholds.cpu().tolist(),
                "model": net_benefit.cpu().tolist(),
                "treat_all": treat_all.cpu().tolist(),
                "treat_none": treat_none.cpu().tolist(),
            },
        }

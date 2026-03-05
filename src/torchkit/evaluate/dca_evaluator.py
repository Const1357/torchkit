from __future__ import annotations

from typing import Any

import torch

from torchkit.evaluate._evaluator import Evaluator, MetricDirection


class DCAEvaluator(Evaluator):
    """
    Decision Curve Analysis (DCA).

    Computes:
        - Net benefit curve
        - Treat-all net benefit
        - Treat-none baseline
        - Best threshold by net benefit

    Expected inputs:
        logits:  (N,2) or (N,)
        targets: (N,) with {0,1}
    """

    def __init__(
        self,
        *,
        logits_key: str,
        target_key: str,
        name: str = "dca",
        primary_metric: str = "max_net_benefit",
        direction: MetricDirection = "maximize",
        weight: float = 1.0,
        n_thresholds: int = 100,
    ):
        super().__init__(
            name=name,
            primary_metric=primary_metric,
            direction=direction,
            weight=weight,
        )

        self.logits_key = logits_key
        self.target_key = target_key
        self.n_thresholds = int(n_thresholds)

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

        N = len(probs)

        thresholds = torch.linspace(0.01, 0.99, self.n_thresholds, device=probs.device)

        net_benefit = []
        treat_all = []

        prevalence = targets.mean()

        for t in thresholds:

            pred_pos = probs >= t

            tp = ((pred_pos) & (targets == 1)).sum().float()
            fp = ((pred_pos) & (targets == 0)).sum().float()

            weight = t / (1 - t)

            nb = (tp / N) - (fp / N) * weight

            net_benefit.append(nb)

            # treat-all strategy
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
            }
        }
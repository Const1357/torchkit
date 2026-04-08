from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from torchkit.evaluate.select._selector_evaluator import SelectorEvaluator
from torchkit.evaluate.mixins._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class _DCASelectorBase(CalibratedLogitsFallbackMixin, SelectorEvaluator):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str,
        direction: str,
        weight: float = 1.0,
        n_thresholds: int = 100,
    ) -> None:
        super().__init__(name=name, direction=direction, weight=weight)
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
    def _binary_positive_probability(x: Tensor) -> Tensor:
        if x.ndim == 2 and x.shape[1] == 2:
            return torch.softmax(x, dim=1)[:, 1]
        if x.ndim == 2 and x.shape[1] == 1:
            return torch.sigmoid(x[:, 0])
        if x.ndim == 1:
            return torch.sigmoid(x)
        raise ValueError(
            f"Expected binary scores/probabilities of shape (N,), (N,1), or (N,2), got {tuple(x.shape)}."
        )

    @staticmethod
    def _positive_probability_from_probabilities(x: Tensor) -> Tensor:
        if x.ndim == 2 and x.shape[1] == 2:
            return x[:, 1]
        if x.ndim == 2 and x.shape[1] == 1:
            return x[:, 0]
        if x.ndim == 1:
            return x
        raise ValueError(
            f"Expected binary probabilities of shape (N,), (N,1), or (N,2), got {tuple(x.shape)}."
        )

    def _net_benefit_curve(self, inputs: dict[str, Any]) -> tuple[Tensor, Tensor]:
        scores = self._resolve_score_tensor(inputs)
        targets = self.resolve(inputs, self.target_key).detach()

        if targets.ndim != 1:
            raise ValueError(f"targets must be shape (N,), got {tuple(targets.shape)}")

        if self.probabilities_key is not None:
            probs_tensor = self.resolve(inputs, self.probabilities_key).detach()
            probs = self._positive_probability_from_probabilities(probs_tensor)
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
        thresholds = torch.linspace(0.01, 0.99, self.n_thresholds, device=probs.device, dtype=probs.dtype)

        net_benefit = []
        for t in thresholds:
            pred_pos = probs >= t
            tp = ((pred_pos) & (targets == 1)).sum().float()
            fp = ((pred_pos) & (targets == 0)).sum().float()
            weight = t / (1 - t)
            nb = (tp / n) - (fp / n) * weight
            net_benefit.append(nb)

        return thresholds, torch.stack(net_benefit)


class MaximumNetBenefitSelectorEvaluator(_DCASelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "max_net_benefit",
        weight: float = 1.0,
        n_thresholds: int = 100,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            name=name,
            direction="maximize",
            weight=weight,
            n_thresholds=n_thresholds,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, net_benefit = self._net_benefit_curve(inputs)
        return torch.max(net_benefit)


class MeanNetBenefitSelectorEvaluator(_DCASelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "mean_net_benefit",
        weight: float = 1.0,
        n_thresholds: int = 100,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            name=name,
            direction="maximize",
            weight=weight,
            n_thresholds=n_thresholds,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, net_benefit = self._net_benefit_curve(inputs)
        return net_benefit.mean()

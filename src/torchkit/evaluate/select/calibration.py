from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from torchkit.evaluate.select._selector_evaluator import SelectorEvaluator
from torchkit.evaluate.mixins._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class _CalibrationSelectorBase(CalibratedLogitsFallbackMixin, SelectorEvaluator):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str,
        direction: str,
        weight: float = 1.0,
        n_bins: int = 15,
    ) -> None:
        super().__init__(name=name, direction=direction, weight=weight)
        self.score_key = score_key
        self.target_key = target_key
        self.probabilities_key = probabilities_key
        self.n_bins = int(n_bins)

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

    def _probs_and_targets(self, inputs: dict[str, Any]) -> tuple[Tensor, Tensor]:
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

        return probs, targets.float()

    def _ece_and_mce(self, probs: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        bin_edges = torch.linspace(0, 1, self.n_bins + 1, device=probs.device)
        ece = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        mce = torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        n = len(probs)

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
            weight = mask.sum() / n

            ece = ece + gap * weight
            mce = torch.maximum(mce, gap)

        return ece, mce


class BrierScoreSelectorEvaluator(_CalibrationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "brier",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            name=name,
            direction="minimize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        probs, targets = self._probs_and_targets(inputs)
        return torch.mean((probs - targets) ** 2)


class ExpectedCalibrationErrorSelectorEvaluator(_CalibrationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "ece",
        weight: float = 1.0,
        n_bins: int = 15,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            name=name,
            direction="minimize",
            weight=weight,
            n_bins=n_bins,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        probs, targets = self._probs_and_targets(inputs)
        ece, _ = self._ece_and_mce(probs, targets)
        return ece


class MaximumCalibrationErrorSelectorEvaluator(_CalibrationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "mce",
        weight: float = 1.0,
        n_bins: int = 15,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            name=name,
            direction="minimize",
            weight=weight,
            n_bins=n_bins,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        probs, targets = self._probs_and_targets(inputs)
        _, mce = self._ece_and_mce(probs, targets)
        return mce
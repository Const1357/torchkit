from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from torchkit.evaluate.select._selector_evaluator import SelectorEvaluator
from torchkit.evaluate.mixins._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class _ROCBinarySelectorBase(CalibratedLogitsFallbackMixin, SelectorEvaluator):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str,
        direction: str,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, direction=direction, weight=weight)
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

    def _roc_tensors(self, inputs: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        scores_tensor = self._resolve_score_tensor(inputs)
        targets = self.resolve(inputs, self.target_key).detach()

        if targets.ndim != 1:
            raise ValueError(f"targets must be shape (N,), got {tuple(targets.shape)}")

        if self.probabilities_key is not None:
            probs_tensor = self.resolve(inputs, self.probabilities_key).detach()
            scores = self._positive_probability_from_probabilities(probs_tensor)
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
                f"{self.__class__.__name__} requires both positive and negative samples. Got positives={int(p.item())}, negatives={int(n.item())}."
            )

        tp = torch.cumsum(pos.to(dtype), dim=0)
        fp = torch.cumsum(neg.to(dtype), dim=0)

        tpr = tp / p
        fpr = fp / n

        zero = torch.zeros(1, device=device, dtype=dtype)
        tpr = torch.cat([zero, tpr])
        fpr = torch.cat([zero, fpr])
        thresholds = torch.cat([torch.tensor([1.0], device=device, dtype=dtype), scores])

        return tpr, fpr, thresholds


class ROCAUCSelectorEvaluator(_ROCBinarySelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "roc_auc",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        from sklearn.metrics import roc_auc_score

        scores_tensor = self._resolve_score_tensor(inputs)
        targets = self.resolve(inputs, self.target_key).detach()

        if targets.ndim != 1:
            raise ValueError(f"targets must be shape (N,), got {tuple(targets.shape)}")

        if self.probabilities_key is not None:
            probs_tensor = self.resolve(inputs, self.probabilities_key).detach()
            scores = self._positive_probability_from_probabilities(probs_tensor)
        else:
            scores = self._binary_positive_probability(scores_tensor)

        if scores.ndim != 1:
            raise ValueError(f"Resolved positive-class scores must be shape (N,), got {tuple(scores.shape)}")
        if scores.shape[0] != targets.shape[0]:
            raise ValueError(
                f"Scores batch size {scores.shape[0]} does not match targets batch size {targets.shape[0]}"
            )

        value = roc_auc_score(
            targets.detach().cpu().numpy(),
            scores.detach().cpu().numpy(),
        )
        return torch.tensor(float(value), device=scores.device, dtype=scores.dtype)


class YoudenJSelectorEvaluator(_ROCBinarySelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "youden_j",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        tpr, fpr, _ = self._roc_tensors(inputs)
        return torch.max(tpr - fpr)


class SensitivityAtYoudenSelectorEvaluator(_ROCBinarySelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "sensitivity_at_youden",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        tpr, fpr, _ = self._roc_tensors(inputs)
        j_stat = tpr - fpr
        j_idx = torch.argmax(j_stat)
        return tpr[j_idx]


class SpecificityAtYoudenSelectorEvaluator(_ROCBinarySelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        name: str = "specificity_at_youden",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        tpr, fpr, _ = self._roc_tensors(inputs)
        j_stat = tpr - fpr
        j_idx = torch.argmax(j_stat)
        return 1.0 - fpr[j_idx]

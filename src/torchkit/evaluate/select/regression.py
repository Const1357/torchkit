from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from torchkit.evaluate.select._selector_evaluator import SelectorEvaluator


class _RegressionSelectorBase(SelectorEvaluator):
    def __init__(
        self,
        *,
        pred_key: str,
        target_key: str,
        name: str,
        direction: str,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, direction=direction, weight=weight)
        self.pred_key = pred_key
        self.target_key = target_key

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.pred_key, self.target_key)

    def _regression_tensors(self, inputs: dict[str, Any]) -> tuple[Tensor, Tensor]:
        preds = self.resolve(inputs, self.pred_key).detach().float()
        targets = self.resolve(inputs, self.target_key).detach().float()

        if preds.ndim == 1:
            preds = preds.unsqueeze(1)
        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have identical shape")

        return preds, targets


class MeanSquaredErrorSelectorEvaluator(_RegressionSelectorBase):
    def __init__(
        self,
        *,
        pred_key: str,
        target_key: str,
        name: str = "mse",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            name=name,
            direction="minimize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets = self._regression_tensors(inputs)
        return ((preds - targets) ** 2).mean()


class RootMeanSquaredErrorSelectorEvaluator(_RegressionSelectorBase):
    def __init__(
        self,
        *,
        pred_key: str,
        target_key: str,
        name: str = "rmse",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            name=name,
            direction="minimize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets = self._regression_tensors(inputs)
        return torch.sqrt(((preds - targets) ** 2).mean())


class MeanAbsoluteErrorSelectorEvaluator(_RegressionSelectorBase):
    def __init__(
        self,
        *,
        pred_key: str,
        target_key: str,
        name: str = "mae",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            name=name,
            direction="minimize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets = self._regression_tensors(inputs)
        return torch.abs(preds - targets).mean()


class R2SelectorEvaluator(_RegressionSelectorBase):
    def __init__(
        self,
        *,
        pred_key: str,
        target_key: str,
        name: str = "r2",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets = self._regression_tensors(inputs)
        target_mean = targets.mean(dim=0)
        ss_tot = ((targets - target_mean) ** 2).sum(dim=0)
        ss_res = ((preds - targets) ** 2).sum(dim=0)
        r2 = 1 - (ss_res / (ss_tot + 1e-12))
        return r2.mean()


class PearsonCorrelationSelectorEvaluator(_RegressionSelectorBase):
    def __init__(
        self,
        *,
        pred_key: str,
        target_key: str,
        name: str = "pearson",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            pred_key=pred_key,
            target_key=target_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets = self._regression_tensors(inputs)
        pred_centered = preds - preds.mean(dim=0)
        target_centered = targets - targets.mean(dim=0)
        corr = (pred_centered * target_centered).sum(dim=0) / (
            torch.sqrt((pred_centered ** 2).sum(dim=0))
            * torch.sqrt((target_centered ** 2).sum(dim=0))
            + 1e-12
        )
        return corr.mean()
from __future__ import annotations

from typing import Any

import torch

from torchkit.evaluate.report._report_evaluator import ReportEvaluator


class RegressionReportEvaluator(ReportEvaluator):
    """
    Regression evaluation.

    Supports multiple targets.

    Expected inputs:
        preds:   (N,) or (N, T)
        targets: (N,) or (N, T)
    """

    def __init__(
        self,
        *,
        pred_key: str,
        target_key: str,
        name: str = "regression",
    ) -> None:
        super().__init__(name=name)

        self.pred_key = pred_key
        self.target_key = target_key

    @property
    def required_keys(self) -> tuple[str, ...]:
        return (self.pred_key, self.target_key)

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        preds = self.resolve(inputs, self.pred_key).detach().float()
        targets = self.resolve(inputs, self.target_key).detach().float()

        if preds.ndim == 1:
            preds = preds.unsqueeze(1)

        if targets.ndim == 1:
            targets = targets.unsqueeze(1)

        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have identical shape")

        _, t = preds.shape

        errors = preds - targets
        abs_errors = torch.abs(errors)
        sq_errors = errors ** 2

        mse = sq_errors.mean(dim=0)
        rmse = torch.sqrt(mse)
        mae = abs_errors.mean(dim=0)

        target_mean = targets.mean(dim=0)
        ss_tot = ((targets - target_mean) ** 2).sum(dim=0)
        ss_res = sq_errors.sum(dim=0)
        r2 = 1 - (ss_res / (ss_tot + 1e-12))

        pred_centered = preds - preds.mean(dim=0)
        target_centered = targets - targets.mean(dim=0)

        corr = (pred_centered * target_centered).sum(dim=0) / (
            torch.sqrt((pred_centered ** 2).sum(dim=0))
            * torch.sqrt((target_centered ** 2).sum(dim=0))
            + 1e-12
        )

        metrics: dict[str, Any] = {
            "mse": float(mse.mean()),
            "rmse": float(rmse.mean()),
            "mae": float(mae.mean()),
            "r2": float(r2.mean()),
            "pearson": float(corr.mean()),
        }

        for target_idx in range(t):
            metrics[f"mse/target_{target_idx}"] = float(mse[target_idx])
            metrics[f"rmse/target_{target_idx}"] = float(rmse[target_idx])
            metrics[f"mae/target_{target_idx}"] = float(mae[target_idx])
            metrics[f"r2/target_{target_idx}"] = float(r2[target_idx])
            metrics[f"pearson/target_{target_idx}"] = float(corr[target_idx])

        return metrics

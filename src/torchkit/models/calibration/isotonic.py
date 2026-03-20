from __future__ import annotations

import torch
from torch import Tensor

from torchkit.models.calibration._calibrator import Calibrator
from torchkit.models._spec_utils import normalize_spec_kwargs


class IsotonicRegressionCalibrator(Calibrator):
    def __init__(self, eps: float = 1e-6, active: bool = False):
        super().__init__(active=active)
        self._spec_kwargs = normalize_spec_kwargs({"eps": eps})

        if not (0.0 < eps < 0.5):
            raise ValueError(f"eps must be in (0, 0.5), got {eps}.")

        self.eps = float(eps)

        self.register_buffer("x_thresholds", torch.empty(0, dtype=torch.float32))
        self.register_buffer("y_thresholds", torch.empty(0, dtype=torch.float32))

    @property
    def is_fitted(self) -> bool:
        return self.x_thresholds.numel() > 0 and self.y_thresholds.numel() > 0

    @staticmethod
    def _extract_binary_score(logits: Tensor) -> Tensor:
        if logits.ndim == 1:
            return logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            return logits[:, 0]
        if logits.ndim == 2 and logits.shape[1] == 2:
            return logits[:, 1] - logits[:, 0]
        raise ValueError(
            f"IsotonicRegressionCalibrator expects binary logits of shape (N,), (N,1), or (N,2), got {tuple(logits.shape)}."
        )

    @staticmethod
    def _restore_binary_shape(score: Tensor, reference_logits: Tensor) -> Tensor:
        if reference_logits.ndim == 1:
            return score
        if reference_logits.ndim == 2 and reference_logits.shape[1] == 1:
            return score.unsqueeze(1)
        if reference_logits.ndim == 2 and reference_logits.shape[1] == 2:
            return torch.stack([-0.5 * score, 0.5 * score], dim=1)
        raise ValueError(
            f"Unsupported binary logit shape: {tuple(reference_logits.shape)}."
        )

    @staticmethod
    def _interp1d(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
        # xp is assumed sorted ascending and length >= 1
        if xp.numel() == 1:
            return torch.full_like(x, fp[0])

        idx = torch.searchsorted(xp, x, right=False)
        idx = torch.clamp(idx, 1, xp.numel() - 1)

        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]

        denom = (x1 - x0).clamp_min(1e-12)
        t = (x - x0) / denom
        y = y0 + t * (y1 - y0)

        y = torch.where(x <= xp[0], fp[0], y)
        y = torch.where(x >= xp[-1], fp[-1], y)
        return y

    def forward_impl(self, logits: Tensor) -> Tensor:
        if not self.is_fitted:
            raise ValueError(f"{self.__class__.__name__} must be fit before calling forward.")

        score = self._extract_binary_score(logits)

        xp = self.x_thresholds.to(score.device)
        fp = self.y_thresholds.to(score.device)

        probs = self._interp1d(score, xp, fp)
        probs = probs.clamp(self.eps, 1.0 - self.eps)

        calibrated_score = torch.logit(probs)
        return self._restore_binary_shape(calibrated_score, logits)

    def fit_impl(self, logits: Tensor, targets: Tensor) -> None:
        from sklearn.isotonic import IsotonicRegression

        if targets.ndim != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects targets of shape (N,), got {tuple(targets.shape)}."
            )

        score = self._extract_binary_score(logits).detach().cpu()
        targets = targets.detach().cpu()

        if score.numel() == 0:
            raise ValueError(f"{self.__class__.__name__} cannot fit on empty logits/targets.")
        if torch.any((targets != 0) & (targets != 1)):
            raise ValueError(f"{self.__class__.__name__} expects binary targets in {{0,1}}.")

        iso = IsotonicRegression(
            y_min=self.eps,
            y_max=1.0 - self.eps,
            out_of_bounds="clip",
            increasing=True,
        )
        iso.fit(score.numpy(), targets.numpy())

        x_thresholds = torch.tensor(iso.X_thresholds_, dtype=torch.float32, device=self.x_thresholds.device)
        y_thresholds = torch.tensor(iso.y_thresholds_, dtype=torch.float32, device=self.y_thresholds.device)

        self.x_thresholds = x_thresholds
        self.y_thresholds = y_thresholds

    def to_spec(self):
        return super().to_spec()

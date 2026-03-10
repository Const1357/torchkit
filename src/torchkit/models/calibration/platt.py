from __future__ import annotations

import torch
from torch import Tensor, nn

from torchkit.models.calibration._calibrator import Calibrator


class PlattScalingCalibrator(Calibrator):
    def __init__(self, init_a: float = 1.0, init_b: float = 0.0, max_iter: int = 100, lr: float = 0.01, active: bool = False):
        super().__init__(active=active)

        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}.")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}.")

        self.a = nn.Parameter(torch.tensor([float(init_a)], dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor([float(init_b)], dtype=torch.float32))
        self.max_iter = int(max_iter)
        self.lr = float(lr)

    @staticmethod
    def _extract_binary_score(logits: Tensor) -> Tensor:
        if logits.ndim == 1:
            return logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            return logits[:, 0]
        if logits.ndim == 2 and logits.shape[1] == 2:
            return logits[:, 1] - logits[:, 0]
        raise ValueError(
            f"PlattScalingCalibrator expects binary logits of shape (N,), (N,1), or (N,2), got {tuple(logits.shape)}."
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

    def forward_impl(self, logits: Tensor) -> Tensor:
        score = self._extract_binary_score(logits)
        calibrated_score = self.a * score + self.b
        return self._restore_binary_shape(calibrated_score, logits)

    def fit_impl(self, logits: Tensor, targets: Tensor) -> None:
        if targets.ndim != 1:
            raise ValueError(
                f"{self.__class__.__name__} expects targets of shape (N,), got {tuple(targets.shape)}."
            )

        score = self._extract_binary_score(logits).detach().to(self.a.device)
        targets = targets.detach().to(self.a.device).float()

        if score.numel() == 0:
            raise ValueError(f"{self.__class__.__name__} cannot fit on empty logits/targets.")
        if torch.any((targets != 0) & (targets != 1)):
            raise ValueError(f"{self.__class__.__name__} expects binary targets in {{0,1}}.")

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.a, self.b], lr=self.lr, max_iter=self.max_iter)

        def closure():
            optimizer.zero_grad()
            calibrated_score = self.a * score + self.b
            loss = criterion(calibrated_score, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
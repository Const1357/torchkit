from __future__ import annotations

import torch
from torch import Tensor, nn

from torchkit.models.calibration._calibrator import Calibrator


class TemperatureScalingCalibrator(Calibrator):
    def __init__(self, init_temp: float = 1.0, max_iter: int = 50, lr: float = 0.01, active: bool = False):
        super().__init__(active=active)

        if init_temp <= 0:
            raise ValueError(f"init_temp must be > 0, got {init_temp}.")
        if max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {max_iter}.")
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}.")

        self.temperature = nn.Parameter(torch.tensor([float(init_temp)], dtype=torch.float32))
        self.max_iter = int(max_iter)
        self.lr = float(lr)

    def forward_impl(self, logits: Tensor) -> Tensor:
        return logits / self.temperature.clamp_min(1e-6)

    def fit_impl(self, logits: Tensor, targets: Tensor) -> None:
        if logits.ndim != 2:
            raise ValueError(f"{self.__class__.__name__} expects multiclass logits of shape (N, C), got {tuple(logits.shape)}.")
        if targets.ndim != 1:
            raise ValueError(f"{self.__class__.__name__} expects targets of shape (N,), got {tuple(targets.shape)}.")
        if logits.shape[1] < 2:
            raise ValueError(f"{self.__class__.__name__} expects at least 2 classes in logits, got shape {tuple(logits.shape)}.")

        logits = logits.detach().to(self.temperature.device)
        targets = targets.detach().to(self.temperature.device).long()

        if targets.numel() == 0:
            raise ValueError(f"{self.__class__.__name__} cannot fit on empty logits/targets.")
        if torch.any(targets < 0) or torch.any(targets >= logits.shape[1]):
            raise ValueError(f"{self.__class__.__name__} expects targets in [0, C-1], got min={int(targets.min())}, max={int(targets.max())}, C={logits.shape[1]}.")

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward_impl(logits), targets)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            self.temperature.clamp_(min=1e-6)
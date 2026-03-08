from __future__ import annotations
from abc import ABC, abstractmethod

from torch import nn, Tensor


class Calibrator(nn.Module, ABC):
    # subclass of nn.Module does not necessarily mean neural network.
    # it gives standard interface: forward, state_dict, to(device) etc.
    # during implementation, if e.g., temperature scaling, have a single parameter and handle its learning during fit.

    # contracts for calibration:
    # - forward takes logits Tensor and returns calibrated logits Tensor of identical shape
    # - fit learns calibration parameters from gathered logits and targets
    def __init__(self, *args, active: bool = True, **kwargs):
        self._active = bool(active)
        super().__init__()

    @property
    def is_active(self) -> bool:
        """Return whether this calibrator is active (i.e., should be used in forward pass)."""
        return self._active
    
    def enable(self) -> "Calibrator":
        """Enable this task calibrator."""
        self._active = True
        return self

    def disable(self) -> "Calibrator":
        """Disable this calibrator."""
        self._active = False
        return self

    def forward(self, logits: Tensor) -> Tensor:

        if not isinstance(logits, Tensor):
            raise ValueError(f"{self.__class__.__name__} forward expects `logits` to be a Tensor. Got {type(logits).__name__} instead.")

        if self.is_active is False:
            return logits  # skip calibration if not active
        
        calibrated_output = self.forward_impl(logits)   # actual logic is called here

        if not isinstance(calibrated_output, Tensor):
            raise ValueError(f"{self.__class__.__name__} forward expects output of `forward_impl` to be a Tensor. Got {type(calibrated_output).__name__} instead.")
        if calibrated_output.shape != logits.shape:
            raise ValueError(f"{self.__class__.__name__} forward expects output of `forward_impl` to have the same shape as input logits. Got {calibrated_output.shape} instead of {logits.shape}.")

        return calibrated_output

    def fit(self, logits: Tensor, targets: Tensor) -> None:

        if not isinstance(logits, Tensor):
            raise ValueError(f"{self.__class__.__name__} fit expects `logits` to be a Tensor. Got {type(logits).__name__} instead.")

        if not isinstance(targets, Tensor):
            raise ValueError(f"{self.__class__.__name__} fit expects `targets` to be a Tensor. Got {type(targets).__name__} instead.")

        if logits.ndim == 0:
            raise ValueError(f"{self.__class__.__name__} fit expects `logits` to be batched. Got scalar Tensor.")
        if targets.ndim == 0:
            raise ValueError(f"{self.__class__.__name__} fit expects `targets` to be batched. Got scalar Tensor.")

        if logits.shape[0] != targets.shape[0]:
            raise ValueError(
                f"{self.__class__.__name__} fit expects `logits` and `targets` to have matching batch size. "
                f"Got {logits.shape[0]} and {targets.shape[0]}."
            )

        self.fit_impl(logits, targets)  # actual logic is called here


    @abstractmethod
    def forward_impl(self, logits: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement `forward_impl` method for applying calibration to logits.")

    @abstractmethod
    def fit_impl(self, logits: Tensor, targets: Tensor) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} must implement `fit_impl` method for learning calibration parameters from logits and targets.")
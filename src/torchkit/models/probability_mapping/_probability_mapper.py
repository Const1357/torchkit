from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn, Tensor

class ProbabilityMapper(nn.Module, ABC):
    """Responsible to transform (optionally calibrated) logits -> probabilities."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, logits: Tensor) -> Tensor:
        if not isinstance(logits, Tensor):
            raise ValueError(f"{self.__class__.__name__} forward expects `logits` to be a Tensor. Got {type(logits).__name__} instead.")

        probs = self.forward_impl(logits)

        if not isinstance(probs, Tensor):
            raise ValueError(f"{self.__class__.__name__} forward expects output of `forward_impl` to be a Tensor. Got {type(probs).__name__} instead.")
        if probs.shape != logits.shape:
            raise ValueError(f"{self.__class__.__name__} forward expects output of `forward_impl` to have the same shape as input logits. Got {probs.shape} instead of {logits.shape}.")

        return probs

    @abstractmethod
    def forward_impl(self, logits: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement `forward_impl` method for transforming logits to probabilities.")
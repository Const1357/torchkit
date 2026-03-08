from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn, Tensor

class DecisionModule(nn.Module, ABC):
    """Responsible to transform probabilities -> predictions."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, probs: Tensor) -> Tensor:
        if not isinstance(probs, Tensor):
            raise ValueError(f"{self.__class__.__name__} forward expects `probs` to be a Tensor. Got {type(probs).__name__} instead.")

        probs = self.forward_impl(probs)

        if not isinstance(probs, Tensor):
            raise ValueError(f"{self.__class__.__name__} forward expects output of `forward_impl` to be a Tensor. Got {type(probs).__name__} instead.")
        
        return probs

    @abstractmethod
    def forward_impl(self, probs: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement `forward_impl` method for transforming probabilities to predictions.")
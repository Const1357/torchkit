from __future__ import annotations

from abc import ABC, abstractmethod

from torch import nn, Tensor

from torchkit.models._spec_utils import resolve_spec_kwargs


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

    @property
    def is_trainable(self) -> bool:
        return type(self).fit_impl is not DecisionModule.fit_impl

    def fit(self, probs: Tensor, targets: Tensor) -> None:
        if not isinstance(probs, Tensor):
            raise ValueError(
                f"{self.__class__.__name__} fit expects `probs` to be a Tensor. "
                f"Got {type(probs).__name__} instead."
            )

        if not isinstance(targets, Tensor):
            raise ValueError(
                f"{self.__class__.__name__} fit expects `targets` to be a Tensor. "
                f"Got {type(targets).__name__} instead."
            )

        if probs.ndim == 0:
            raise ValueError(f"{self.__class__.__name__} fit expects `probs` to be batched. Got scalar Tensor.")
        if targets.ndim == 0:
            raise ValueError(f"{self.__class__.__name__} fit expects `targets` to be batched. Got scalar Tensor.")

        if probs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"{self.__class__.__name__} fit expects `probs` and `targets` to have matching batch size. "
                f"Got {probs.shape[0]} and {targets.shape[0]}."
            )

        self.fit_impl(probs, targets)

    @abstractmethod
    def forward_impl(self, probs: Tensor) -> Tensor:
        raise NotImplementedError(f"{self.__class__.__name__} must implement `forward_impl` method for transforming probabilities to predictions.")

    def fit_impl(self, probs: Tensor, targets: Tensor) -> None:
        return None

    def to_spec(self):
        from torchkit.models.decision.factory import DecisionModuleSpec

        return DecisionModuleSpec(
            cls=self.__class__,
            kwargs=resolve_spec_kwargs(self),
        )

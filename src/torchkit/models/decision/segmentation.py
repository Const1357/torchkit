from __future__ import annotations

from torch import Tensor
import torch

from torchkit.models.decision._decision_module import DecisionModule


class BinarySegmentationThreshold(DecisionModule):
    """
    Binary segmentation decision module.

    Supported input shapes:
    - (B, 1, *spatial) : binary probabilities
    - (B, *spatial)    : binary probabilities (no channel dim)

    Returns:
    - (B, 1, *spatial) integer predictions in {0, 1}
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}.")

        self._threshold = float(threshold)

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}.")
        self._threshold = float(value)

    def forward_impl(self, probs: Tensor) -> Tensor:
        if probs.ndim < 3:
            raise ValueError(
                f"{self.__class__.__name__} expects segmentation probabilities of shape "
                f"(B, 1, *spatial) or (B, *spatial). Got {tuple(probs.shape)}."
            )

        # Case: (B, 1, *spatial)
        if probs.ndim >= 4 and probs.shape[1] == 1:
            p_pos = probs

        # Case: (B, *spatial) → no channel dim
        elif probs.ndim >= 3:
            p_pos = probs.unsqueeze(1)

        else:
            raise ValueError(
                f"{self.__class__.__name__} received invalid shape {tuple(probs.shape)}."
            )

        return (p_pos >= self.threshold).to(dtype=torch.long)
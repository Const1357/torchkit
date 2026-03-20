from __future__ import annotations

from torch import Tensor
import torch

from torchkit.models.probability_mapping._probability_mapper import ProbabilityMapper

class SegmentationProbabilityMapper(ProbabilityMapper):
    """
    Supports binary and multiclass segmentation logits.

    Expected input shapes:
      - Binary segmentation:    (B, 1, *spatial)
      - Multiclass segmentation:(B, C, *spatial), where C >= 2

    Output:
      - Binary: sigmoid probabilities with same shape as input
      - Multiclass: softmax probabilities over channel dim=1, same shape as input
    """

    def forward_impl(self, mask: Tensor) -> Tensor:
        if mask.ndim < 3:
            raise ValueError(
                f"{self.__class__.__name__} expects segmentation logits of shape "
                f"(B, 1, *spatial) for binary or (B, C, *spatial) for multiclass, "
                f"with at least one spatial dimension. Got shape {tuple(mask.shape)}."
            )

        n_channels = mask.shape[1]

        if n_channels == 1:
            # Binary segmentation logits: (B, 1, *spatial)
            return torch.sigmoid(mask)

        if n_channels >= 2:
            # Multiclass segmentation logits: (B, C, *spatial)
            return torch.softmax(mask, dim=1)

        raise ValueError(
            f"{self.__class__.__name__} received invalid channel dimension {n_channels} "
            f"for input shape {tuple(mask.shape)}."
        )

    def to_spec(self):
        return super().to_spec()

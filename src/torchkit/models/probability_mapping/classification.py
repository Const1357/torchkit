from __future__ import annotations

from torch import Tensor
import torch

from torchkit.models.probability_mapping._probability_mapper import ProbabilityMapper


class ClassificationProbabilityMapper(ProbabilityMapper):
    """ Supports Binary and Multiclass classification.
     - For binary classification, applies thresholding to sigmoid probabilities.
     - For multiclass classification, applies argmax to softmax probabilities.
    Supports shapes: (N,) (N, 1), (N, C>=2). Output shape is identical to input shape.
      
    ### *Note* 
    A (N, 2) shape corresponds to binary classification but will be softmax-activated.
    """

    def forward_impl(self, logits: Tensor) -> Tensor:

        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            # binary logits of shape (N,) or (N, 1)
            probs = torch.sigmoid(logits)
            return probs

        if logits.ndim == 2 and logits.shape[1] >= 2:
            # multiclass logits of shape (N, C >=2)
            probs = torch.softmax(logits, dim=1)
            return probs

        raise ValueError(
            f"{self.__class__.__name__} expects binary or multiclass logits of shape (N,), (N,1), or (N,C) with C>=2. "
            f"Got shape {tuple(logits.shape)}."
        )

    def to_spec(self):
        return super().to_spec()

from __future__ import annotations

from torch import Tensor
import torch

from torchkit.models.decision._decision_module import DecisionModule


class BinaryClassificationThreshold(DecisionModule):
    """
    Binary classification decision module.

    Supported input shapes:
    - (N,)   : binary probabilities
    - (N, 1) : binary probabilities
    - (N, 2) : two-class probabilities

    Returns:
    - (N,) integer predictions in {0, 1}

    ### *Note*
    For (N, 2) input shape, the second column (index 1) is treated
        as the positive class probability.
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

        if probs.ndim == 1:
            p_pos = probs

        elif probs.ndim == 2 and probs.shape[1] == 1:
            p_pos = probs[:, 0]

        elif probs.ndim == 2 and probs.shape[1] == 2:
            p_pos = probs[:, 1]

        else:
            raise ValueError(
                f"{self.__class__.__name__} expects binary probabilities of shape "
                f"(N,), (N,1), or (N,2). Got shape {tuple(probs.shape)}."
            )

        return (p_pos >= self.threshold).to(dtype=torch.long)
    

class ArgmaxDecision(DecisionModule):
    """
    Multiclass classification decision module.

    Supported input shapes:
    - (N, C) with C >= 2 : multiclass probabilities

    Returns:
    - (N,) integer predictions in {0, 1, ..., C-1}
    """

    def forward_impl(self, probs: Tensor) -> Tensor:

        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError(
                f"{self.__class__.__name__} expects multiclass probabilities of shape (N, C) with C >= 2. "
                f"Got shape {tuple(probs.shape)}."
            )

        return torch.argmax(probs, dim=1)
    
class SampleTopKTemperature(DecisionModule):
    """
    Multiclass classification decision module that samples from top-k classes with temperature scaling.

    Supported input shapes:
    - (N, C) with C >= 2 : multiclass probabilities

    Returns:
    - (N,) integer predictions in {0, 1, ..., C-1}
    """

    def __init__(self, k: int = 5, temperature: float = 1.0):
        super().__init__()

        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        self.k = int(k)
        self.temperature = float(temperature)

    def forward_impl(self, probs: Tensor) -> Tensor:

        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError(
                f"{self.__class__.__name__} expects multiclass probabilities of shape (N, C) with C >= 2. "
                f"Got shape {tuple(probs.shape)}."
            )

        # Apply temperature scaling
        scaled_probs = torch.pow(probs, 1.0 / self.temperature)

        # Get top-k indices
        topk_probs, topk_indices = torch.topk(scaled_probs, k=min(self.k, probs.shape[1]), dim=1)

        # Normalize top-k probabilities
        topk_probs_normalized = topk_probs / torch.sum(topk_probs, dim=1, keepdim=True)

        # Sample from the top-k distribution
        sampled_indices = torch.multinomial(topk_probs_normalized, num_samples=1).squeeze(1)

        # Map back to original class indices
        return topk_indices.gather(1, sampled_indices.unsqueeze(1)).squeeze(1)
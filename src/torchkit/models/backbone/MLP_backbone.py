from __future__ import annotations
from typing import Any, Collection, Optional

from torch import nn, Tensor
from torchkit.models.backbone._backbone import Backbone
from torchkit.models._spec_utils import normalize_spec_kwargs

class MLPBackbone(Backbone):
    """A simple MLP backbone that produces a single feature map. Serves as a minimal example and a template for custom backbones."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        *,
        activation: nn.Module = nn.ReLU(),
        norm: nn.Module | None = None,
        dropout: float = 0.0,
    ):
        super().__init__(supported_features=["features"])
        self._spec_kwargs = normalize_spec_kwargs(
            {
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "output_dim": output_dim,
                "activation": activation,
                "norm": norm,
                "dropout": dropout,
            }
        )

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if norm is not None:
                layers.append(norm(hidden_dim))
            layers.append(activation)
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def _forward_impl(self, input: dict[str, Any], requested_features: Optional[Collection[str]] = None) -> dict[str, Tensor]:
        x = input.get("x")  # convention: backbone input is always under key "x"
        features = self.mlp(x)
        return {"features": features}

    def to_spec(self):
        return super().to_spec()

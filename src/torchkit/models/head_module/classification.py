from __future__ import annotations

from torch import nn, Tensor


class ClassifierHeadLinear(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.n_output = num_classes

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        logits = self.linear(x)
        return {"logits": logits}


class ClassifierHeadMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        *,
        activation: nn.Module = nn.ReLU,
        norm: nn.Module | None = None,  # eg. nn.BatchNorm1d, InstanceNorm1d, LayerNorm, etc. Applied after linear and before activation.
        dropout: float = 0.0,
    ):
        super().__init__()

        if len(hidden_dims) == 0:
            raise ValueError(
                "ClassifierHeadMLP requires at least one hidden layer. "
                "Use ClassifierHeadLinear for a single linear classifier."
            )

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if norm is not None:
                layers.append(norm(hidden_dim))

            layers.append(activation())

            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.n_output = num_classes
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
    ) -> dict[str, Tensor]:

        logits = self.mlp(x)

        return {"logits": logits}
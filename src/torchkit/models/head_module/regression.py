from __future__ import annotations

from torch import nn, Tensor

from torchkit.models._spec_utils import normalize_spec_kwargs


class RegressorHeadLinear(nn.Module):

    def __init__(self, input_dim: int, n_targets: int = 1):
        super().__init__()
        self._spec_kwargs = normalize_spec_kwargs(
            {
                "input_dim": input_dim,
                "n_targets": n_targets,
            }
        )
        if n_targets <= 0:
            raise ValueError(f"`n_targets` must be positive, got {n_targets}.")
        self.linear = nn.Linear(input_dim, n_targets)
        self.n_output = int(n_targets)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # Output shape: (B, n_targets)
        return {f"predictions": self.linear(x)}

    def to_spec(self):
        from torchkit.models.head_module.factory import HeadModuleSpec

        return HeadModuleSpec(
            cls=self.__class__,
            kwargs=normalize_spec_kwargs(self._spec_kwargs),
        )


class RegressorHeadMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        n_targets: int = 1,
        *,
        activation: nn.Module = nn.ReLU,
        norm: nn.Module | None = None,  # e.g., nn.BatchNorm1d, nn.LayerNorm, etc. Applied after linear and before activation.
        dropout: float = 0.0,
    ):
        super().__init__()
        self._spec_kwargs = normalize_spec_kwargs(
            {
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "n_targets": n_targets,
                "activation": activation,
                "norm": norm,
                "dropout": dropout,
            }
        )

        if len(hidden_dims) == 0:
            raise ValueError(
                "RegressorHeadMLP requires at least one hidden layer. "
                "Use RegressorHeadLinear for a single linear regressor."
            )
        if n_targets <= 0:
            raise ValueError(f"`n_targets` must be positive, got {n_targets}.")
        if dropout < 0.0:
            raise ValueError(f"`dropout` must be >= 0.0, got {dropout}.")

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

        layers.append(nn.Linear(prev_dim, n_targets))

        self.n_output = int(n_targets)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # Output shape: (B, n_targets)
        return {f"predictions": self.mlp(x)}

    def to_spec(self):
        from torchkit.models.head_module.factory import HeadModuleSpec

        return HeadModuleSpec(
            cls=self.__class__,
            kwargs=normalize_spec_kwargs(self._spec_kwargs),
        )

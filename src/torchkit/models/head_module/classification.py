from __future__ import annotations
from typing import Optional

from torch import nn, Tensor

from torchkit.models._spec_utils import normalize_spec_kwargs


class ClassifierHeadLinear(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self._spec_kwargs = normalize_spec_kwargs(
            {
                "input_dim": input_dim,
                "num_classes": num_classes,
            }
        )
        self.linear = nn.Linear(input_dim, num_classes)
        self.n_output = num_classes

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        logits = self.linear(x)
        return {"logits": logits}

    def to_spec(self):
        from torchkit.models.head_module.factory import HeadModuleSpec

        return HeadModuleSpec(
            cls=self.__class__,
            kwargs=normalize_spec_kwargs(self._spec_kwargs),
        )


class ClassifierHeadMLP(nn.Module):
    """Lazy MLP head for classification tasks.
    
    ### *Important*
    If you use a it as a lazy module, you must ensure that
    the lazy layer is initialized before instantiating a `Trainer` by passing a dummy input."""

    def __init__(
        self,
        hidden_dims: list[int],
        num_classes: int,
        *,
        input_dim: Optional[int],
        activation: nn.Module = nn.ReLU,
        norm: nn.Module | None = None,  # eg. nn.BatchNorm1d, InstanceNorm1d, LayerNorm, etc. Applied after linear and before activation.
        dropout: float = 0.0,
    ):
        super().__init__()
        self._spec_kwargs = normalize_spec_kwargs(
            {
                "hidden_dims": hidden_dims,
                "num_classes": num_classes,
                "input_dim": input_dim,
                "activation": activation,
                "norm": norm,
                "dropout": dropout,
            }
        )

        if len(hidden_dims) == 0:
            raise ValueError(
                "ClassifierHeadMLP requires at least one hidden layer. "
                "Use ClassifierHeadLinear for a single linear classifier."
            )

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):

            if i == 0 and prev_dim is None:
                layers.append(nn.LazyLinear(hidden_dim))
            else:
                if prev_dim is None:
                    raise ValueError("prev_dim should have been inferred after the first lazy layer.")
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

    def to_spec(self):
        from torchkit.models.head_module.factory import HeadModuleSpec

        return HeadModuleSpec(
            cls=self.__class__,
            kwargs=normalize_spec_kwargs(self._spec_kwargs),
        )

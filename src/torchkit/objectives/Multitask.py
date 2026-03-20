from __future__ import annotations
from typing import Any, Literal
import warnings

try:
    from typing import override  # py3.12+
except ImportError:
    from typing_extensions import override  # py<=3.11

import torch
from torch import Tensor

from torchkit.objectives._base import Objective


class MultitaskObjective(Objective):
    """Weighted sum of multiple objectives (each with its own weight).
    Per objective loss can be found in `self.per_objective_loss` after forward/loss. It resets every call."""

    def __init__(
        self,
        *objectives: Objective | list[Objective] | tuple[Objective, ...],
        name: str,
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        is_optional: bool = False,
    ):
        if len(objectives) == 1 and isinstance(objectives[0], (list, tuple)):
            objs = list(objectives[0])
        else:
            objs = list(objectives)

        if not objs:
            raise ValueError("At least one objective must be provided.")

        for i, obj in enumerate(objs):
            if not isinstance(obj, Objective):
                raise TypeError(
                    f"All objectives must derive from Objective, got {type(obj).__name__} at index {i}."
                )

        if (not is_optional) and all(obj.is_optional for obj in objs):
            raise ValueError(
                "MultitaskObjective is required but all contained objectives are optional."
            )

        super().__init__(name=name, weight=weight, is_optional=is_optional, reduction=reduction)
        self._objectives: tuple[Objective, ...] = tuple(objs)
        if not all(obj.reduction == self.reduction for obj in self._objectives):
            raise ValueError(
                f"All contained objectives must have the same reduction as the MultitaskObjective. "
                f"Expected reduction {self.reduction}, but got {[obj.reduction for obj in self._objectives]}."
            )

        # diagnostics container (populated on forward/loss)
        self.per_objective_loss: dict[str, float | list[float]] = {}

    @property
    def objectives(self) -> tuple[Objective, ...]:
        return self._objectives

    @property
    def required_keys(self) -> tuple[str, ...]:
        # Union of required paths across contained objectives
        keys: set[str] = set()
        for obj in self._objectives:
            keys.update(obj.required_keys)
        return tuple(sorted(keys))

    @override
    def loss(self, *, inputs: dict[str, Any]) -> Tensor:
        total_loss: Tensor | None = None
        self.per_objective_loss = {}

        for obj in self._objectives:
            obj_loss = obj(inputs=inputs)
            weighted = obj.weight * obj_loss

            # diagnostics (detach)
            with torch.no_grad():
                if obj_loss.ndim == 0:
                    self.per_objective_loss[obj.name] = float(obj_loss.detach().cpu().item())
                else:
                    self.per_objective_loss[obj.name] = obj_loss.detach().cpu().flatten().tolist()

            total_loss = weighted if total_loss is None else (total_loss + weighted)

        if total_loss is None:
            if self.is_optional:
                warnings.warn(
                    f"Multitask objective {self.name!r} is optional but no objectives contributed. "
                    f"Per-objective losses: {self.per_objective_loss}. Returning zero loss."
                )
                return self._zero_loss(inputs)
            raise RuntimeError(
                f"Multitask objective {self.name!r} is required but no objectives contributed. "
                f"Per-objective losses: {self.per_objective_loss}."
            )

        return total_loss

    @override
    def forward(
        self,
        *,
        inputs: dict[str, Any] | None,
    ) -> Tensor:
        if not self._check_call_arguments(inputs=inputs):
            return self._zero_loss(inputs={})

        assert inputs is not None
        # No explicit missing-key checks: each objective handles missing/optional logic.
        return self.loss(inputs=inputs)

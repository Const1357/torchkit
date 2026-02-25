from __future__ import annotations
from typing import override

from torch import Tensor
import torch

from torchkit.objectives._base import Objective

import warnings

class MultitaskObjective(Objective):
    """Interface class for Multitask objectives."""

    def __init__(
        self,
        *objectives: Objective | list[Objective] | tuple[Objective, ...],
        name: str,
        weight: float = 1.0,
        is_optional: bool = False,
    ):
        
        if len(objectives) == 1 and isinstance(objectives[0], (list, tuple)):
            objs = list(objectives[0])
        else:
            objs = list(objectives)

        if not objs:
            raise ValueError(f"At least one objective must be provided. Got {len(objs)}.")
        
        for i, obj in enumerate(objs):
            if not isinstance(obj, Objective):
                raise TypeError(
                    f"All objectives must be derived from `Objective`, got type {type(obj)} at index {i}."
                )
            
        if not is_optional and all(obj.is_optional for obj in objs):
            raise ValueError("At least one objective must be required (not optional) if the multitask objective is required.")

        # required keys are the union of all contained objectives' required keys
        required_keys = set().union(*(obj.required_keys for obj in objs))

        super().__init__(
            name=name,
            weight=weight,
            is_optional=is_optional,
            required_keys=required_keys,
        )

        self._objectives: tuple[Objective, ...] = tuple(objs)

    @property
    def objectives(self) -> tuple[Objective, ...]:
        return self._objectives
    
    
    # --- public API ---
    @override
    def loss(self, inputs: dict[str, Tensor], reduction: str = "mean") -> Tensor:
        
        total_loss: torch.Tensor = None  # type: ignore[assignment]

        per_objective_loss = dict[str, Tensor] = {}

        for obj in self.objectives:
            obj_loss: Tensor = obj(inputs, reduction=reduction)
            weighted_loss = obj.weight * obj_loss

            per_objective_loss[obj.name] = obj_loss.item() if obj_loss.ndim == 0 else obj_loss.tolist() # diagnostics

            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss


        if total_loss is None:

            if self.is_optional:
                warnings.warn(f"Multitask objective '{self.name}' is optional but no objectives contributed to the total loss. Per-objective (unweighted) losses: {per_objective_loss}. Returning zero loss.")
                return self._zero_loss(inputs)

            raise RuntimeError(f"Multitask objective '{self.name}' is required but no objectives contributed to the total loss. Per-objective (unweighted) losses: {per_objective_loss}.")
        
        return total_loss



    # interface
    @override
    def __call__(self, inputs: dict[str, Tensor], reduction: str = "mean") -> Tensor:
        
        self._check_call_arguments(inputs=inputs, reduction=reduction)
        
        # no missing key checks here. If keys are missing, objectives will skip or raise depending or their own optional status.

        return self.loss(inputs, reduction=reduction)
        
        



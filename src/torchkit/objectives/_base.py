from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch import nn, Tensor


class Objective(nn.Module, ABC):
    """
    Base class for Objectives.

    An Objective consumes a nested mapping of model outputs and computes a scalar loss.

    Design contract:
    - `forward(inputs=...)` receives a nested dict whose leaves are Tensors.
    - Derived classes define string paths (e.g. "clf/logits", "reg/predictions")
    that specify where required tensors live inside the nested mapping.
    - Paths are resolved using "/" as a namespace separator.
    - **Derived classes MUST implement**:
        - `required_keys`: tuple of required path strings.
        - `loss(...)`: actual loss computation using `Objective.resolve(...)`.

    The base class:
    - Validates inputs and reduction.
    - Checks required paths.
    - Handles optional objectives by returning a zero loss if inputs are missing.
    - Provides path resolution via `resolve(...)`.
    """

    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        is_optional: bool = False,
    ):
        
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Objective `reduction` must be one of 'mean', 'sum', got {reduction}.")
        if not isinstance(is_optional, bool):
            raise TypeError(f"Objective `is_optional` must be a boolean, got {type(is_optional).__name__}.")

        super().__init__()

        # bookkeeping
        self._name = self._validate_name(name)
        self._weight = self._validate_weight(weight)
        self._reduction = reduction
        self._is_optional = bool(is_optional)

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError(f"Objective `name` must be a string, got {type(name).__name__}.")
        if not name:
            raise ValueError(f"Objective `name` must be a non-empty string, got {name}.")
        return str(name)

    @staticmethod
    def _validate_weight(weight: float) -> float:
        if weight < 0.0:
            raise ValueError(f"Objective `weight` must be non-negative, got {weight}.")
        return float(weight)

    @property
    def name(self) -> str:
        """The name of the objective."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = self._validate_name(value)

    @property
    def weight(self) -> float:
        """The weight of the objective."""
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = self._validate_weight(value)

    @property
    def is_optional(self) -> bool:
        """Whether the objective is optional."""
        return self._is_optional
    
    @property
    def required_keys(self) -> tuple[str, ...]:
        raise NotImplementedError(f"Objective derived class {self.__class__.__name__} must define the `required_keys` property.")
    
    @property
    def reduction(self) -> Literal["mean", "sum"]:
        """The reduction method to apply to the output loss tensor."""
        return self._reduction

    # --- internal helpers ---
    @staticmethod
    def resolve(inputs: dict[str, Any], key: str) -> Tensor:
        current: Any = inputs
        parts = [part for part in key.split("/") if part]

        for i, part in enumerate(parts):

            path = "/".join(parts[:i]) or "<root>"

            if not isinstance(current, dict):
                raise TypeError(f"Expected a dict at path {path}, but got {type(current).__name__}.")
            if part not in current:
                raise KeyError(f"Key {part!r} not found at path {path}. Available keys: {list(current.keys())}.")
            current = current[part]

        if current is None:
            raise ValueError(f"Resolved value for key {key!r} is None. All required keys must be present and non-None.")
        if not isinstance(current, Tensor):
            raise TypeError(f"Resolved value for key {key!r} must be a Tensor, got {type(current).__name__}.")
        return current

    def _missing_keys(self, inputs: dict[str, Any] | None) -> list[str]:
        """Return the list of required paths that cannot be resolved to a non-None Tensor."""
        if not self.required_keys:
            return []
        if inputs is None:
            return list(self.required_keys)

        missing: list[str] = []
        for key in self.required_keys:
            try:
                _ = self.resolve(inputs, key)
            except (KeyError, TypeError, ValueError):
                missing.append(key)
        return missing
    
    def _zero_loss(self, inputs: dict[str, Any]) -> Tensor:
        """
        Return a zero loss for optional objectives when required keys are missing. 
        Device and dtype are ideally inferred from the first available floating-point 
        tensor. If only non-float tensors are present, the device is inferred from 
        the first available tensor, and dtype defaults to the global default.\\
        Keys containing "label", "mask", or "target" (case-insensitive) are heuristically skipped when searching for a tensor to infer device/dtype,
        as these are likely to be supervision tensors (e.g., class labels, binary masks).
        """
        device = torch.device("cpu")
        dtype = torch.get_default_dtype()
        found_any_tensor = False

        if self.required_keys:
            for key in self.required_keys:
                
                if "label" in key.lower() or "mask" in key.lower() or "target" in key.lower():
                    continue  # HACK: skip keys that likely correspond to supervision tensors (heuristic assumming conventional naming)

                try: tensor = self.resolve(inputs, key)
                except (KeyError, TypeError, ValueError): continue

                # Save the very first tensor's device as a fallback, 
                # just in case there are no floating-point tensors at all.
                if not found_any_tensor:
                    device = tensor.device
                    found_any_tensor = True
                
                # If we find a float tensor, grab BOTH its device and dtype, then stop.
                # hunting for a float tensor mitigates the risk of capturing a supervision tensor (e.g., label, binary mask etc.)
                if tensor.is_floating_point():
                    device = tensor.device
                    dtype = tensor.dtype
                    break

        return torch.tensor(0.0, device=device, dtype=dtype, requires_grad=True)
    
    def _check_call_arguments(
        self,
        inputs: dict[str, Any] | None,
    ) -> bool:
        
        if inputs is None:
            if self.is_optional:
                return False  # returning false => should return zero loss.
            raise TypeError(f"Required Objective '{self.name}' inputs must not be None.")
        if not isinstance(inputs, dict):
            raise TypeError(f"Objective inputs must be a dict, got {type(inputs).__name__}.")
        if not all(isinstance(key, str) for key in inputs.keys()):
            raise TypeError("All keys in the inputs dict must be strings.")
        
        return True  # returning true => should proceed to loss calculation

        # We will not recursively typecheck the `inputs` variable that it either contains nested dicts
        # or in the base case a Tensor. These checks happen in `resolve` and are lazily deferred.

    # --- public API ---
    @abstractmethod
    def loss(self, *, inputs: dict[str, Any]) -> Tensor:
        """Calculate the loss for the objective.\\
            Use Objective.resolve to resolve names from the inputs dict and obtain named Tensors as instructed during construction.
            
            (During construction of the instance, your internal variables will be given a path to be resolved by the resolver.)"""
        raise NotImplementedError(f"Objective derived class {self.__class__.__name__} must implement the `loss` method.")

    # interface: perform checks and delegate to `loss` method
    def forward(self, *, inputs: dict[str, Any] | None,) -> Tensor:

        if not self._check_call_arguments(inputs=inputs):
            return self._zero_loss(inputs={})  # type: ignore

        missing_keys = self._missing_keys(inputs)

        if missing_keys:

            if self.is_optional:
                # If the objective is optional, return a zero loss if keys are missing
                return self._zero_loss(inputs)
            
            raise KeyError(f"Objective '{self.name}' is missing required keys: {missing_keys}. Provided keys (top-level): {list(inputs.keys())}.")

        loss = self.loss(inputs=inputs)
        if not isinstance(loss, Tensor):
            raise TypeError(f"Objective '{self.name}' `loss` method must return a Tensor, got {type(loss).__name__}.")
        if loss.ndim != 0:
            raise ValueError(f"Objective '{self.name}' `loss` method must return a scalar (0-dim) Tensor, got {loss.ndim}-dim Tensor.")
        
        return loss

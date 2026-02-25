from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import Tensor

class Objective(ABC):
    """Base class for objectives.\\
        Provides a routing interface (dict-like) for capturing\
        the requirements for the objective calculation from the forward pass of the model."""
    
    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        is_optional: bool = False,
        required_keys: tuple[str, ...] | list[str] = (),
    ):
        
        if not isinstance(name, str):
            raise TypeError(f"Objective `name` must be a string, got {type(name)}.")
        if not name:
            raise ValueError(f"Objective `name` must be a non-empty string, got {name}.")
        if weight < 0.0:
            raise ValueError(f"Objective `weight` must be non-negative, got {weight}.")
        if not isinstance(is_optional, bool):
            raise TypeError(f"Objective `is_optional` must be a boolean, got {type(is_optional)}.")
        if not isinstance(required_keys, (tuple, list)):
            raise TypeError(f"Objective `required_keys` must be a tuple or list of strings, got {type(required_keys)}.")
        if not all(isinstance(key, str) for key in required_keys):
            raise TypeError("All elements in `required_keys` must be strings.")
        if len(set(required_keys)) != len(required_keys):
            raise ValueError(f"All elements in `required_keys` must be unique. Provided: {required_keys} contains duplicates.")
        
        self._name = str(name)
        self._weight = float(weight)
        self._is_optional = bool(is_optional)
        self._required_keys = tuple(required_keys)

    @property
    def name(self) -> str:
        """The name of the objective."""
        return self._name

    @property
    def weight(self) -> float:
        """The weight of the objective."""
        return self._weight

    @property
    def is_optional(self) -> bool:
        """Whether the objective is optional."""
        return self._is_optional

    @property
    def required_keys(self) -> tuple[str, ...]:
        """The required keys for the objective."""
        return self._required_keys
    
    # --- internal helpers ---
    def _missing_keys(self, inputs: dict[str, Tensor]) -> list[str]:
        """Check for missing keys in the inputs. Missing means either the key is not present or the value is None.\\
            ### NOTE: The objective designer should handle the case of nans or infs in the loss calculation if all keys are present."""

        if not self.required_keys: return []
        if inputs is None: return list(self.required_keys)

        return [key for key in self.required_keys if key not in inputs or inputs[key] is None]
    
    def _zero_loss(self, inputs: dict[str, Tensor]) -> Tensor:
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

                tensor = inputs.get(key)
                
                if tensor is not None:
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
        inputs: dict[str, Tensor] | None,
        reduction: str,
    ) -> None:
        
        if not isinstance(reduction, str):
            raise TypeError(f"Objective reduction must be a string, got {type(reduction)}.")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Objective reduction must be one of 'mean', 'sum', or 'none', got {reduction}.")
        
        if inputs is None:
            if self.is_optional:
                return self._zero_loss({})
            raise TypeError(f"Required Objective '{self.name}' inputs must not be None.")
        if not isinstance(inputs, dict):
            raise TypeError(f"Objective inputs must be a dict, got {type(inputs)}.")
        if not all(isinstance(key, str) for key in inputs.keys()):
            raise TypeError("All keys in the inputs dict must be strings.")
        if not all(isinstance(value, Tensor) or value is None for value in inputs.values()):
            raise TypeError("All values in the inputs dict must be torch.Tensor or None.")


    # --- public API ---
    @abstractmethod
    def loss(self, *, inputs: dict[str, Tensor], reduction: str = "mean") -> Tensor:
        """Calculate the loss for the objective.\\
            The `inputs` dict will contain all the required keys specified in `required_keys`."""
        pass

    # interface: perform checks and delegate to `loss` method
    def __call__(self, *, inputs: dict[str, Tensor] | None, reduction: Literal["mean", "sum", "none"] = "mean") -> Tensor:

        self._check_call_arguments(inputs=inputs, reduction=reduction)

        missing_keys = self._missing_keys(inputs)

        if missing_keys:

            if self.is_optional:
                # If the objective is optional, we can return a zero loss if keys are missing
                return self._zero_loss(inputs)
            
            raise KeyError(f"Objective '{self.name}' is missing required keys: {missing_keys}. Provided keys: {list(inputs.keys())}.")

        return self.loss(inputs=inputs, reduction=reduction)
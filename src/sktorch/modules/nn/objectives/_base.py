from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Literal

from torch import Tensor


@dataclass
class LossOut():
    """
    Output of an objective loss computation.
    + loss (torch.Tensor): Scalar tensor loss to be **minimized**.
    + details (Dict[str, Any]): Additional details about the loss (e.g., per-sample losses, metrics, etc.).
    """
    loss: Tensor
    details: Dict[str, Any]


class _BaseObjective(ABC):

    def __init__(
        self,
        *,
        name: str,
        objective_type: Literal['supervised', 'unsupervised', 'multitask'] = 'supervised',
        weight: float = 1.0,
        required: bool=True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
    ):
        if not name:
            raise ValueError(f"Objective name must be a non-empty string, got {name}.")
        if weight <= 0.0:
            raise ValueError(f"Objective weight must be positive (> 0), got {weight}.")
        if objective_type not in ('supervised', 'unsupervised', 'multitask'):
            raise ValueError(f"Objective type must be in ['supervised', 'unsupervised', 'multitask'], got {objective_type}.")
        if not isinstance(required_pred_keys, (tuple, list)):
            raise TypeError(f"required_pred_keys must be a list or tuple of strings, got type {type(required_pred_keys)}.")
        if not isinstance(required_target_keys, (tuple, list)):
            raise TypeError(f"required_target_keys must be a list or tuple of strings, got type {type(required_target_keys)}.")
        
        for k in required_pred_keys:
            if not isinstance(k, str) or not k:
                raise ValueError(f"required_pred_keys must contain non-empty strings, got {k!r} ({type(k)}).")
        for k in required_target_keys:
            if not isinstance(k, str) or not k:
                raise ValueError(f"required_target_keys must contain non-empty strings, got {k!r} ({type(k)}).")
        
        self._name = name
        self._type = objective_type
        self._weight = float(weight)
        self._required = bool(required)
        self._required_pred_keys = tuple(required_pred_keys)
        self._required_target_keys = tuple(required_target_keys)

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def weight(self) -> float:
        return self._weight
    
    @property
    def required(self) -> bool:
        return self._required
    
    @property
    def required_pred_keys(self) -> tuple[str, ...]:
        return self._required_pred_keys
    
    @property
    def required_target_keys(self) -> tuple[str, ...]:
        return self._required_target_keys
    
    def _zero_loss(self, predictions: Mapping[str, Tensor | None], targets: Mapping[str, Tensor | None] | None) -> Tensor:
        """
        Return a scalar zero loss tensor that is device/dtype-safe and graph-connected."""

        # only need predictions to construct graph-connected zero-loss
        for tensor in predictions.values():
            if tensor is not None:
                return tensor.sum() * 0.0       # graph-connected zero tensor

        raise RuntimeError(f"Cannot construct graph-connected zero-loss: No tensors found in predictions.")
    
    def _check_required_keys(self, predictions: Mapping[str, Tensor | None], targets: Mapping[str, Tensor | None] | None) -> tuple[bool, Dict[str, Any]]:

        info: dict[str, Any] = {}

        missing_pred_keys = [key for key in self._required_pred_keys if predictions.get(key) is None]
        if missing_pred_keys:
            if self._required:
                raise KeyError(f"Missing required prediction keys {missing_pred_keys} for objective '{self._name}'.")
            else:
                info['missing_pred_keys'] = missing_pred_keys
                return False, info
            
        if self._type == 'supervised':
            if targets is None:
                if self._required:
                    raise ValueError(f"Targets must be provided for supervised objective '{self._name}'. Got {targets}.")
                else:
                    info['missing_all_target_keys'] = list(self._required_target_keys)
                    return False, info
            
            missing_target_keys = [key for key in self._required_target_keys if targets.get(key) is None]
            if missing_target_keys:
                if self._required:
                    raise KeyError(f"Missing required target keys {missing_target_keys} for objective '{self._name}'.")
                else:
                    info['missing_target_keys'] = missing_target_keys
                    return False, info
                
        return True, info
    
    @abstractmethod
    def _loss(self, predictions: Mapping[str, Tensor | None], targets: Mapping[str, Tensor | None] | None) -> LossOut:
        """Unified **internal** entry point."""
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"weight={self._weight}, "
            f"required={self._required}, "
            f"required_pred_keys={self._required_pred_keys}, "
            f"required_target_keys={self._required_target_keys})"
            f")"
        )
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, weight={self._weight}, type={self._type})"



class SupervisedObjective(_BaseObjective):
    """Base class for supervised learning objectives.
    """

    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        required: bool = True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
    ):
        
        if not required_target_keys:
            raise ValueError(f"SupervisedObjective requires at least one required_target_key, got {required_target_keys}.")

        super().__init__(
            name=name,
            objective_type='supervised',
            weight=weight,
            required=required,
            required_pred_keys=required_pred_keys,
            required_target_keys=required_target_keys,
        )

    @abstractmethod
    def loss(self, predictions: Mapping[str, Tensor], targets: Mapping[str, Tensor]) -> LossOut:
        """Compute the supervised loss given predictions and targets.

        Args:
            predictions (Mapping[str, Tensor]): Model predictions.
            targets (Mapping[str, Tensor]): Ground truth targets."""
        ...
        
    def _loss(self, predictions: Mapping[str, Tensor | None], targets: Mapping[str, Tensor | None] | None) -> LossOut:
            
        valid, info = self._check_required_keys(predictions, targets)
        if not valid:
            return LossOut(loss=self._zero_loss(predictions, targets), details=info)
        
        assert targets is not None    # sanity
        return self.loss(predictions, targets)
    

class UnsupervisedObjective(_BaseObjective):
    """Base class for supervised learning objectives.
    """
    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        required: bool = True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
    ):
        super().__init__(
            name=name,
            objective_type='unsupervised',
            weight=weight,
            required=required,
            required_pred_keys=required_pred_keys,
            required_target_keys=(),
        )
    
    @abstractmethod
    def loss(self, predictions: Mapping[str, Tensor]) -> LossOut:
        """Compute the unsupervised loss given predictions.

        Args:
            predictions (Mapping[str, Tensor]): Model predictions.
        """
        ...

    def _loss(self, predictions: Mapping[str, Tensor | None], targets: Mapping[str, Tensor | None] | None) -> LossOut:
        
        # ignore targets for unsupervised objectives
        valid, info = self._check_required_keys(predictions, None)
        if not valid:
            return LossOut(loss=self._zero_loss(predictions, None), details=info)

        return self.loss(predictions)
        

class MultitaskObjective(_BaseObjective):
    """Mix multiple objectives into a single multitask objective.
    """

    def __init__(
        self,
        *objectives: _BaseObjective | list[_BaseObjective] | tuple[_BaseObjective, ...],
        required: bool = True,
    ):
        if len(objectives) == 1 and isinstance(objectives[0], (list, tuple)):
            objs = list(objectives[0])
        else:
            objs = list(objectives)

        if not objs:
            raise ValueError(f"At least one objective must be provided. Got {len(objs)}.")
        
        for i, obj in enumerate(objs):
            if not isinstance(obj, _BaseObjective):
                raise TypeError(f"All objectives must be derived from _BaseObjective, got type {type(obj)} at index {i}.")
            
        if all(obj.required == False for obj in objs) and required:
            raise ValueError("At least one objective must be required if the multitask objective is required.")
        
        self._objectives: tuple[_BaseObjective, ...] = tuple(objs)

        super().__init__(
            name=f"Mix:{'+'.join(f'{obj.weight}*{obj.name}' for obj in self._objectives)}",
            objective_type='multitask',
            required=required,
            required_pred_keys=tuple(sorted(set().union(*(obj.required_pred_keys for obj in self._objectives)))),
            required_target_keys=tuple(sorted(set().union(*(obj.required_target_keys for obj in self._objectives)))),
        )

    @property
    def objectives(self) -> tuple[_BaseObjective, ...]:
        return self._objectives
    
    def _loss(self, predictions: Mapping[str, Tensor | None], targets: Mapping[str, Tensor | None] | None) -> LossOut:
        total: Tensor | None = None
        details: Dict[str, Any] = {}

        for obj in self._objectives:

            obj_loss_out = obj._loss(predictions, targets)  # invoke internal _loss

            l = obj_loss_out.loss
            d = obj_loss_out.details

            if l.ndim != 0:
                raise ValueError(f"Objective loss must be a scalar tensor (shape [ ]), got shape {l.shape} from objective {obj.name}.")

            weighted_loss = obj.weight * l
            total = weighted_loss if total is None else total + weighted_loss

            details[f"{obj.name}/loss"] = l.detach()
            details[f"{obj.name}/weighted_loss"] = weighted_loss.detach()
            for key, value in d.items():    # flattened dictionary
                details[f"{obj.name}/{key}"] = value

        assert total is not None    # sanity
        return LossOut(loss=total, details=details)
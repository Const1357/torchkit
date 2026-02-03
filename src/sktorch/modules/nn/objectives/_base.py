from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Literal

try:
    from typing import final    # Python 3.11+
except ImportError:
    from typing_extensions import final

from torch import Tensor

# NOTE: any changes here should also be done in the typehint of _BaseObjective on objective_type.
__OBJECTIVE_TYPES = ('relational', 'intrinsic', 'contextual', 'composite')


@dataclass
class LossOut():
    """
    Output of an objective loss computation.
    + loss (torch.Tensor): Scalar tensor loss to be **minimized**.
    + details (Dict[str, Any]): Additional details about the loss (e.g., per-sample losses, metrics, etc.).
    """
    loss: Tensor
    details: Dict[str, Any] = field(default_factory=dict)


class _BaseObjective(ABC):

    def __init__(
        self,
        *,
        name: str,
        objective_type: Literal['relational', 'intrinsic', 'contextual', 'composite'] = 'relational',   # NOTE: keep in sync with __OBJECTIVE_TYPES
        weight: float = 1.0,
        required: bool=True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
        required_context_keys: tuple[str, ...] | list[str] = (),
    ):  
        if not isinstance(name, str):
            raise TypeError(f"Objective name must be a string, got type {type(name)}.")
        elif not name:
            raise ValueError(f"Objective name must be a non-empty string, got {name}.")
        if weight <= 0.0:
            raise ValueError(f"Objective {name} weight must be positive (> 0), got {weight}.")
        if not isinstance(required, bool):
            raise TypeError(f"Objective {name} required flag must be a boolean, got type {type(required)}.")
        if not isinstance(objective_type, str):
            raise TypeError(f"Objective {name} type must be a string, got type {type(objective_type)}.")
        elif objective_type not in __OBJECTIVE_TYPES:
            raise ValueError(f"Objective {name} type must be in {__OBJECTIVE_TYPES}, got {objective_type}.")
                
        
        # in base class, all required keys must be lists or tuples of non-empty strings. None is not allowed, but this is handled internally from interface->base.
        if not isinstance(required_pred_keys, (tuple, list)):
            raise TypeError(f"Objective {name} required_pred_keys must be a list or tuple of strings, got type {type(required_pred_keys)}.")
        if not isinstance(required_target_keys, (tuple, list)):
            raise TypeError(f"Objective {name} required_target_keys must be a list or tuple of strings, got type {type(required_target_keys)}.")
        if not isinstance(required_context_keys, (tuple, list)):
            raise TypeError(f"Objective {name} required_context_keys must be a list or tuple of strings, got type {type(required_context_keys)}.")

        for k in required_pred_keys:
            if not isinstance(k, str) or not k:
                raise ValueError(f"Objective {name} required_pred_keys must contain non-empty strings, got {k!r} ({type(k)}).")
        for k in required_target_keys:
            if not isinstance(k, str) or not k:
                raise ValueError(f"Objective {name} required_target_keys must contain non-empty strings, got {k!r} ({type(k)}).")
        for k in required_context_keys:
            if not isinstance(k, str) or not k:
                raise ValueError(f"Objective {name} required_context_keys must contain non-empty strings, got {k!r} ({type(k)}).")
            
        self._name = name
        self._type = objective_type
        self._weight = float(weight)
        self._required = bool(required)
        self._required_pred_keys = tuple(required_pred_keys)
        self._required_target_keys = tuple(required_target_keys)
        self._required_context_keys = tuple(required_context_keys)

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
    
    @property
    def required_context_keys(self) -> tuple[str, ...]:
        return self._required_context_keys
    
    def _zero_loss(self, predictions: Mapping[str, Tensor | None]) -> Tensor:
        """
        Return a scalar zero-loss tensor that is device/dtype-safe and graph-connected.
        Anchors ONLY off prediction tensors (never targets/context).
        """
        for v in predictions.values():
            if v is None:
                continue
            if not isinstance(v, Tensor):
                raise TypeError(f"predictions must map str -> Tensor|None, got {type(v)}.")
            return v.nansum() * 0.0

        raise RuntimeError("Cannot construct graph-connected zero-loss: No tensors found in predictions.")
    
    def _check_required_keys(
        self,
        predictions: Mapping[str, Tensor | None] | None,
        targets: Mapping[str, Tensor | None] | None,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Validate required keys for this objective.

        Semantics:
        - Predictions are optional; objectives may be context-only or target-only(unlikely but allowed) in terms of inputs.
        - Missing required keys:
            * if objective is required -> raise immediately.
            * if objective is optional -> return (False, info) ONLY if we can later create a
            graph-connected zero loss from *predictions* (i.e., at least one Tensor exists in predictions).
            Otherwise raise, because a zero loss would not be valid in this system.
        - Context/targets are never used to construct the graph-connected zero loss anchor.
        """

        if predictions is not None and not isinstance(predictions, Mapping):
            raise TypeError(f"predictions must be a Mapping[str, Tensor|None] or None, got {type(predictions)}.")
        if targets is not None and not isinstance(targets, Mapping):
            raise TypeError(f"targets must be a Mapping[str, Tensor|None] or None, got {type(targets)}.")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError(f"context must be a Mapping[str, Any] or None, got {type(context)}.")

        info: dict[str, Any] = {}

        # Helper: do we have any tensor to anchor zero-loss in predictions?
        def _has_pred_tensor_anchor(preds: Mapping[str, Tensor | None] | None) -> bool:
            if preds is None:
                return False
            return any(isinstance(t, Tensor) for t in preds.values())
        
        can_skip_with_zero = _has_pred_tensor_anchor(predictions)

        # predictions keys
        if self._required_pred_keys:
            if predictions is None:
                missing_pred_keys = list(self._required_pred_keys)
            else:
                missing_pred_keys = [k for k in self._required_pred_keys if predictions.get(k) is None]

            if missing_pred_keys:
                if self._required:
                    raise KeyError(
                        f"Missing required prediction keys {missing_pred_keys} for objective '{self._name}'."
                    )
                info["missing_pred_keys"] = missing_pred_keys
                if can_skip_with_zero:
                    return False, info
                raise RuntimeError(
                    f"Objective '{self._name}' is optional but cannot be skipped safely: "
                    f"missing prediction keys {missing_pred_keys} and no prediction tensor exists "
                    f"to construct a graph-connected zero loss."
                )

        # target keys
        if self._required_target_keys:
            if targets is None:
                missing_target_keys = list(self._required_target_keys)
            else:
                missing_target_keys = [k for k in self._required_target_keys if targets.get(k) is None]

            if missing_target_keys:
                if self._required:
                    raise KeyError(
                        f"Missing required target keys {missing_target_keys} for objective '{self._name}'."
                    )
                info["missing_target_keys"] = missing_target_keys
                if can_skip_with_zero:
                    return False, info
                raise RuntimeError(
                    f"Objective '{self._name}' is optional but cannot be skipped safely: "
                    f"missing target keys {missing_target_keys} and no prediction tensor exists "
                    f"to construct a graph-connected zero loss."
                )

        # context keys
        if self._required_context_keys:
            if context is None:
                missing_context_keys = list(self._required_context_keys)
            else:
                # note: context values may be non-tensors; treat None as missing
                missing_context_keys = [k for k in self._required_context_keys if context.get(k) is None]

            if missing_context_keys:
                if self._required:
                    raise KeyError(
                        f"Missing required context keys {missing_context_keys} for objective '{self._name}'."
                    )
                info["missing_context_keys"] = missing_context_keys
                if can_skip_with_zero:
                    return False, info
                raise RuntimeError(
                    f"Objective '{self._name}' is optional but cannot be skipped safely: "
                    f"missing context keys {missing_context_keys} and no prediction tensor exists "
                    f"to construct a graph-connected zero loss."
                )

        return True, info
    
    def _postprocess(self, out: LossOut) -> LossOut:
        """verify LossOut correctness before returning it"""
        if out.loss.ndim != 0:
            raise ValueError(
                f"Objective '{self._name}' must return a scalar loss ([]) tensor (ndim=0), got shape {tuple(out.loss.shape)}."
            )
        if not isinstance(out.details, dict):
            raise TypeError(
                f"Objective '{self._name}' must return details as dict, got {type(out.details)}."
            )
        return out
    
    def _skip_or_raise(
        self,
        predictions: Mapping[str, Tensor | None] | None,
        targets: Mapping[str, Tensor | None] | None,
        context: Mapping[str, Any] | None = None,
    ) -> LossOut | None:
        """
        Returns LossOut with zero-loss if objective should be skipped (optional & missing keys),
        else returns None meaning 'continue and compute real loss'.
        Raises if required keys missing or cannot anchor zero-loss.
        """
        valid, info = self._check_required_keys(predictions, targets, context=context)
        if valid:
            return None
        assert predictions is not None  # otherwise _check_required_keys would have raised
        return LossOut(loss=self._zero_loss(predictions), details=info)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"weight={self._weight}, "
            f"required={self._required}, "
            f"required_pred_keys={self._required_pred_keys}, "
            f"required_target_keys={self._required_target_keys},"
            f"required_context_keys={self._required_context_keys}"
            f")"
        )
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, weight={self._weight}, type={self._type})"

    # loss method to be implemented by objectives
    # __call__ calls loss but with checks. <- this is the final user interface. loss is the internal interface.


class RelationalObjective(_BaseObjective):
    """
    Base class for relational objectives.

    A relational objective defines a loss between predictions and targets,
    optionally conditioned on additional context.

    Typical use cases include classification losses, regression losses,
    contrastive objectives, or any term that explicitly\\
    compares model outputs
    against reference values.

    ### Call signature:
        ```
        obj(predictions, targets, context=None) -> LossOut
        ```

    ### Parameters:
    - **name** (`str`):  
    Human-readable identifier for the objective. Used for logging, diagnostics,
    and composite objective naming.

    - **weight** (`float`, default=`1.0`):  
    Scalar weight applied to the loss when used inside a composite objective.

    - **required** (`bool`, default=`True`):  
    If `True`, missing required inputs will raise immediately.  
    If `False`, the objective may be skipped and replaced with a zero-loss
    (only if a graph-connected zero-loss can be constructed).

    - **required_pred_keys** (`tuple[str, ...] | list[str]`):  
    Keys that must be present in the `predictions` mapping passed to `__call__`.

    - **required_target_keys** (`tuple[str, ...] | list[str]`):  
    Keys that must be present in the `targets` mapping passed to `__call__`.

    - **required_context_keys** (`tuple[str, ...] | list[str]`):  
    Optional keys that must be present in the `context` mapping, if provided.

    ### Example:
    ```
    # Define a relational objective
    ce_loss = CrossEntropyLoss(
        name="cross_entropy",
        weight=1.0,
        required=True,
        required_pred_keys=("clf/logits",),
        required_target_keys=("clf/labels",),
    )

    # Compute the loss
    out = ce_loss(
        predictions={"clf/logits": logits},
        targets={"clf/labels": labels},
    )

    scalar_loss = out.loss
    ```
    """

    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        required: bool = True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
        required_context_keys: tuple[str, ...] | list[str] = ()
    ):
        
        if not required_pred_keys:
            raise ValueError(f"RelationalObjective {name} requires at least one required_pred_key, got {required_pred_keys}.")
        if not required_target_keys:
            raise ValueError(f"RelationalObjective {name} requires at least one required_target_key, got {required_target_keys}.")

        super().__init__(
            name=name,
            objective_type='relational',
            weight=weight,
            required=required,
            required_pred_keys=required_pred_keys,
            required_target_keys=required_target_keys,
            required_context_keys=required_context_keys,
        )

    @abstractmethod
    def loss(
        self, 
        predictions: Mapping[str, Tensor | None],
        targets: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None
    ) -> LossOut:
        ...

    @final 
    def __call__(
        self, 
        predictions: Mapping[str, Tensor | None],
        targets: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None
    ) -> LossOut:
        # override call for interface purposes
        
        skipped = self._skip_or_raise(predictions, targets, context=context)
        if skipped is not None:
            return skipped
        
        out = self.loss(predictions, targets, context=context)
        return self._postprocess(out)
    

class IntrinsicObjective(_BaseObjective):
    """
    Base class for intrinsic objectives.

    An intrinsic objective computes a loss solely from model predictions,
    optionally using contextual information, without requiring external targets.

    These objectives typically act as regularizers or auxiliary signals derived
    from the model’s own outputs, such as entropy terms, consistency penalties,
    or confidence shaping.

    ### Call signature:
    ```
    obj(predictions, context=None) -> LossOut
    ```
    ### Parameters:
    - **name** (`str`):  
    Human-readable identifier for the objective.

    - **weight** (`float`, default=`1.0`):  
    Scalar weight applied to the loss when used inside a composite objective.

    - **required** (`bool`, default=`True`):  
    If `True`, missing required prediction keys will raise.  
    If `False`, the objective may be skipped if a graph-connected zero-loss
    can be constructed from predictions.

    - **required_pred_keys** (`tuple[str, ...] | list[str]`):  
    Keys that must be present in the `predictions` mapping.

    - **required_context_keys** (`tuple[str, ...] | list[str]`):  
    Optional contextual keys required by the objective.
    
    ### Example:
    ```
    # Define an intrinsic objective
    entropy_term = EntropyTerm(
        name="entropy_regularization",
        weight=0.01,
        required=True,
        required_pred_keys=("clf/probs",),
    )

    # Compute the loss
    out = entropy_term(
        predictions={"clf/probs": probs},
    )

    scalar_loss = out.loss
    ```
    """
    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        required: bool = True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_context_keys: tuple[str, ...] | list[str] = (),
    ):  
        
        if not required_pred_keys:
            raise ValueError(f"IntrinsicObjective {name} requires at least one required_pred_key, got {required_pred_keys}.")
        
        super().__init__(
            name=name,
            objective_type='intrinsic',
            weight=weight,
            required=required,
            required_pred_keys=required_pred_keys,
            required_target_keys=(),
            required_context_keys=required_context_keys,
        )

    @abstractmethod
    def loss(
        self, 
        predictions: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None
    ) -> LossOut:
        ...

    @final
    def __call__(
        self, predictions: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None
    ) -> LossOut:
        # override call for interface purposes
        skipped = self._skip_or_raise(predictions, targets=None, context=context)
        if skipped is not None:
            return skipped
        
        out = self.loss(predictions, context=context)
        return self._postprocess(out)
    

class ContextualObjective(_BaseObjective):
    """
    Base class for contextual objectives.

    A contextual objective computes a loss primarily from contextual inputs,
    such as model parameters, optimizer state, or external metadata.
    Predictions and targets may be used if relevant, but are not required.

    Typical use cases include weight decay, parameter norm penalties,
    or constraint-based regularization.

    ### Call signature:
    ```
    obj(context, predictions=None, targets=None) -> LossOut
    ```

    ### Parameters:
    - **name** (`str`):  
    Human-readable identifier for the objective.

    - **weight** (`float`, default=`1.0`):  
    Scalar weight applied to the loss when used inside a composite objective.

    - **required** (`bool`, default=`True`):  
    If `True`, missing required context keys will raise immediately.

    - **required_context_keys** (`tuple[str, ...] | list[str]`):  
    Keys that must be present in the `context` mapping passed to `__call__`.

    - **required_pred_keys** (`tuple[str, ...] | list[str]`):  
    Optional prediction keys required if the objective depends on model outputs.

    - **required_target_keys** (`tuple[str, ...] | list[str]`):  
    Optional target keys required if the objective depends on reference values.

    ### Example:
    ```
    # Define a contextual objective
    l2_penalty = L2Penalty(
        name="l2_weight_decay",
        weight=1e-4,
        required=True,
        required_context_keys=("params",),
    )

    # Compute the loss
    out = l2_penalty(
        context={"params": model.parameters()},
    )

    scalar_loss = out.loss
    ```
    """
    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        required: bool = True,
        required_context_keys: tuple[str, ...] | list[str] = (),
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
    ):
        
        if not required_context_keys:
            raise ValueError(f"ContextualObjective {name} requires at least one required_context_key, got {required_context_keys}.")

        super().__init__(
            name=name,
            objective_type='contextual',
            weight=weight,
            required=required,
            required_pred_keys=required_pred_keys,
            required_target_keys=required_target_keys,
            required_context_keys=required_context_keys,
        )

    @abstractmethod
    def loss(
        self, 
        context: Mapping[str, Any],
        predictions: Mapping[str, Tensor | None] | None = None,
        targets: Mapping[str, Tensor | None] | None = None,
    ) -> LossOut:
        ...

    @final
    def __call__(
        self, 
        context: Mapping[str, Any],
        predictions: Mapping[str, Tensor | None] | None = None,
        targets: Mapping[str, Tensor | None] | None = None,
    ) -> LossOut:
        
        # override call for interface purposes
        skipped = self._skip_or_raise(predictions, targets, context=context)
        if skipped is not None:
            return skipped
        
        out = self.loss(context, predictions=predictions, targets=targets)
        return self._postprocess(out)
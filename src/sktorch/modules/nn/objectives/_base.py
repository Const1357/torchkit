from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

try:
    from typing import final  # Python 3.11+
except ImportError:
    from typing_extensions import final

import torch
from torch import Tensor


@dataclass
class LossOut:
    """
    Output of an objective loss computation.

    + loss (torch.Tensor): Tensor loss to be **minimized**. May be scalar or non-scalar
      (reduction is objective-defined).
    + details (Dict[str, Any]): Additional details about the loss (e.g., per-term info, metrics, etc.).
    """
    loss: Tensor
    details: Dict[str, Any] = field(default_factory=dict)


class _BaseObjective(ABC):
    """
    Base class for objectives.

    Objectives declare required keys from:
    + predictions (Mapping[str, Tensor|None])
    + targets     (Mapping[str, Tensor|None])
    + context     (Mapping[str, Any])

    Contract (simple + user-friendly):
    + A required key is considered missing if:
        - the key does not exist, OR
        - the key exists but its value is None.
      (i.e., required means "must be present and have a real value".)

    Required / optional behavior:
    + required=True  -> missing required keys raise (full report).
    + required=False -> missing required keys skip if a zero-loss can be constructed,
                       otherwise raise.

    Zero-loss semantics (when skipped):
    + Prefer a graph-connected zero anchored to a required prediction tensor (if available).
    + Otherwise anchor to any prediction tensor (if available).
    + Otherwise use a disconnected scalar zero on best-effort device and floating dtype.
      If no tensors exist anywhere to infer device, skipping is not possible and we raise.
    """

    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        required: bool = True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
        required_context_keys: tuple[str, ...] | list[str] = (),
    ):
        if not isinstance(name, str):
            raise TypeError(f"Objective name must be a string, got type {type(name)}.")
        if not name:
            raise ValueError(f"Objective name must be a non-empty string, got {name}.")
        if weight < 0.0:
            raise ValueError(f"Objective {name} weight must be non-negative (>= 0), got {weight}.")
        if not isinstance(required, bool):
            raise TypeError(f"Objective {name} required flag must be a boolean, got type {type(required)}.")

        for nm, keys in (
            ("required_pred_keys", required_pred_keys),
            ("required_target_keys", required_target_keys),
            ("required_context_keys", required_context_keys),
        ):
            if not isinstance(keys, (tuple, list)):
                raise TypeError(f"Objective {name} {nm} must be a list or tuple of strings, got type {type(keys)}.")
            if any((not isinstance(k, str)) for k in keys):
                raise TypeError(f"Objective {name} {nm} must contain only strings, got {keys!r}.")
            if len(set(keys)) != len(keys):
                raise ValueError(f"Objective {name} {nm} contains duplicates: {keys!r}.")

        self._name = name
        self._weight = float(weight)
        self._required = required
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

    def requirements(self) -> Dict[str, tuple[str, ...]]:
        """
        Return required keys for this objective (trainer-facing introspection).

        Returned dict:
        + "pred": required prediction keys
        + "target": required target keys
        + "context": required context keys
        """
        return {
            "pred": self._required_pred_keys,
            "target": self._required_target_keys,
            "context": self._required_context_keys,
        }

    # ---- internal helpers ----

    @staticmethod
    def _validate_tensor_mapping(name: str, m: Mapping[str, Any] | None) -> None:
        """
        Validate that a mapping is Mapping[str, Tensor|None].

        Strict policy:
        + Every value must be either torch.Tensor or None.
        """
        if m is None:
            return
        if not isinstance(m, Mapping):
            raise TypeError(f"{name} must be a Mapping[str, Tensor|None] or None, got {type(m)}.")

        for k, v in m.items():
            if not isinstance(k, str):
                raise TypeError(f"{name} keys must be str, got key {k!r} of type {type(k)}.")
            if v is not None and not isinstance(v, Tensor):
                raise TypeError(f"{name}[{k!r}] must be a torch.Tensor or None, got {type(v)}.")

    @staticmethod
    def _missing_keys(mapping: Mapping[str, Any] | None, required_keys: tuple[str, ...]) -> list[str]:
        """
        Missing means: key does not exist OR key exists but value is None.
        """
        if not required_keys:
            return []
        if mapping is None:
            return list(required_keys)

        missing: list[str] = []
        for k in required_keys:
            if k not in mapping or mapping[k] is None:
                missing.append(k)
        return missing

    @staticmethod
    def _graph_connected_zero_from_tensor(t: Tensor) -> Tensor:
        """
        Return a graph-connected scalar zero anchored to `t`.

        - Connected to autograd via `t.sum()`.
        - Same device/dtype as `t` (for floating tensors).
        - If `t` is non-floating, cast to float32 for a valid autograd anchor.
        """
        if not t.is_floating_point():
            t = t.float()
        return t.sum() * t.new_zeros(())

    @staticmethod
    def _first_tensor_in_mapping(m: Mapping[str, Any] | None) -> Tensor | None:
        if m is None:
            return None
        for v in m.values():
            if isinstance(v, Tensor):
                return v
        return None

    @staticmethod
    def _first_floating_tensor_in_mapping(m: Mapping[str, Any] | None) -> Tensor | None:
        if m is None:
            return None
        for v in m.values():
            if isinstance(v, Tensor) and v.is_floating_point():
                return v
        return None

    def _fallback_ref_tensor(
        self,
        *,
        predictions: Mapping[str, Any] | None,
        targets: Mapping[str, Any] | None,
        context: Mapping[str, Any] | None,
    ) -> Tensor | None:
        """
        Pick a tensor reference to set device/dtype for disconnected fallback zeros.

        Order:
        + predictions (any tensor)
        + targets (any tensor)
        + context (any tensor)
        """
        t = self._first_tensor_in_mapping(predictions)
        if t is not None:
            return t
        t = self._first_tensor_in_mapping(targets)
        if t is not None:
            return t
        t = self._first_tensor_in_mapping(context)
        if t is not None:
            return t
        return None

    def _fallback_float_dtype(
        self,
        *,
        predictions: Mapping[str, Any] | None,
        targets: Mapping[str, Any] | None,
        context: Mapping[str, Any] | None,
    ) -> torch.dtype:
        """
        Choose a floating dtype for disconnected fallback zeros.

        Prefer the first floating tensor dtype from:
        + predictions -> targets -> context
        Otherwise fall back to float32.
        """
        t = self._first_floating_tensor_in_mapping(predictions)
        if t is not None:
            return t.dtype
        t = self._first_floating_tensor_in_mapping(targets)
        if t is not None:
            return t.dtype
        t = self._first_floating_tensor_in_mapping(context)
        if t is not None:
            return t.dtype
        return torch.float32

    def _zero_loss(
        self,
        *,
        predictions: Mapping[str, Any] | None,
        targets: Mapping[str, Any] | None = None,
        context: Mapping[str, Any] | None = None,
        anchor_pred_keys: tuple[str, ...] = (),
    ) -> Tensor:
        """
        Return a zero-loss tensor.

        Preference order:
        1) Graph-connected zero anchored to a required prediction key (first Tensor found).
        2) Graph-connected zero anchored to any prediction Tensor (first Tensor found).
        3) Disconnected scalar zero on best-effort device and floating dtype.

        If no tensor exists anywhere to infer device/dtype, raise.
        """
        # 1) Prefer anchoring on required prediction keys (required-first)
        if predictions is not None and anchor_pred_keys:
            for k in anchor_pred_keys:
                if k in predictions:
                    v = predictions[k]
                    if isinstance(v, Tensor):
                        return self._graph_connected_zero_from_tensor(v)

        # 2) Fall back to any prediction tensor
        if predictions is not None:
            for v in predictions.values():
                if isinstance(v, Tensor):
                    return self._graph_connected_zero_from_tensor(v)

        # 3) Disconnected fallback (needs a tensor somewhere to infer device)
        ref = self._fallback_ref_tensor(predictions=predictions, targets=targets, context=context)
        if ref is None:
            raise RuntimeError(
                f"Cannot construct a zero-loss for objective '{self._name}': "
                f"no prediction tensor to anchor autograd, and no tensors in targets/context "
                f"to infer device/dtype."
            )

        dtype = self._fallback_float_dtype(predictions=predictions, targets=targets, context=context)
        return torch.zeros((), device=ref.device, dtype=dtype)

    def _skip_or_raise(
        self,
        *,
        predictions: Mapping[str, Tensor | None] | None,
        targets: Mapping[str, Tensor | None] | None,
        context: Mapping[str, Any] | None = None,
        zero_anchor_pred_keys: tuple[str, ...] = (),
    ) -> LossOut | None:
        # strict validation for tensors (as requested)
        self._validate_tensor_mapping("predictions", predictions)
        self._validate_tensor_mapping("targets", targets)
        if context is not None and not isinstance(context, Mapping):
            raise TypeError(f"context must be a Mapping[str, Any] or None, got {type(context)}.")

        missing_pred = self._missing_keys(predictions, self._required_pred_keys)
        missing_target = self._missing_keys(targets, self._required_target_keys)
        missing_context = self._missing_keys(context, self._required_context_keys)

        if not missing_pred and not missing_target and not missing_context:
            return None

        info: Dict[str, Any] = {}
        if missing_pred:
            info["missing_pred_keys"] = missing_pred
        if missing_target:
            info["missing_target_keys"] = missing_target
        if missing_context:
            info["missing_context_keys"] = missing_context

        if self._required:
            parts: list[str] = []
            if missing_pred:
                parts.append(f"prediction keys {missing_pred}")
            if missing_target:
                parts.append(f"target keys {missing_target}")
            if missing_context:
                parts.append(f"context keys {missing_context}")
            msg = ", ".join(parts)
            raise KeyError(f"Missing required {msg} for objective '{self._name}'.")

        # optional: skip if able, otherwise raise
        try:
            z = self._zero_loss(
                predictions=predictions,
                targets=targets,
                context=context,
                anchor_pred_keys=zero_anchor_pred_keys,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"Optional objective '{self._name}' cannot be skipped because a zero-loss cannot be constructed. "
                f"Original error: {e}"
            ) from e

        return LossOut(loss=z, details=info)

    def _postprocess(self, out: LossOut) -> LossOut:
        """
        Verify LossOut correctness before returning it.

        Enforced invariants:
        - out is a LossOut
        - out.loss is a torch.Tensor (scalar or non-scalar is allowed)
        - out.details is a dict
        """
        if not isinstance(out, LossOut):
            raise TypeError(f"Objective '{self._name}' must return LossOut, got {type(out)}.")
        if not isinstance(out.loss, Tensor):
            raise TypeError(f"Objective '{self._name}' must return loss as Tensor, got {type(out.loss)}.")
        if out.loss.numel() == 0:
            raise ValueError(f"Objective '{self._name}' returned an empty loss tensor.")
        if not isinstance(out.details, dict):
            raise TypeError(f"Objective '{self._name}' must return details as dict, got {type(out.details)}.")
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"weight={self._weight}, "
            f"required={self._required}, "
            f"required_pred_keys={self._required_pred_keys}, "
            f"required_target_keys={self._required_target_keys}, "
            f"required_context_keys={self._required_context_keys}"
            f")"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, weight={self._weight})"


class RelationalObjective(_BaseObjective):
    """
    Base class for relational objectives.

    Relational objectives compute loss by comparing:
    + predictions against targets
    + optional context (e.g., masks, weights, metadata)
    """

    def __init__(
        self,
        *,
        name: str,
        weight: float = 1.0,
        required: bool = True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
        required_context_keys: tuple[str, ...] | list[str] = (),
    ):
        if not required_pred_keys:
            raise ValueError(
                f"RelationalObjective {name} requires at least one required_pred_key, got {required_pred_keys}."
            )
        if not required_target_keys:
            raise ValueError(
                f"RelationalObjective {name} requires at least one required_target_key, got {required_target_keys}."
            )

        super().__init__(
            name=name,
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
        context: Mapping[str, Any] | None = None,
    ) -> LossOut:
        ...

    @final
    def __call__(
        self,
        predictions: Mapping[str, Tensor | None],
        targets: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> LossOut:
        skipped = self._skip_or_raise(
            predictions=predictions,
            targets=targets,
            context=context,
            zero_anchor_pred_keys=self._required_pred_keys,  # required-first anchoring
        )
        if skipped is not None:
            return skipped
        out = self.loss(predictions, targets, context=context)
        return self._postprocess(out)


class IntrinsicObjective(_BaseObjective):
    """
    Base class for intrinsic objectives.

    Intrinsic objectives compute loss from:
    + predictions (required)
    + optional context
    Targets are not used.
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
            raise ValueError(
                f"IntrinsicObjective {name} requires at least one required_pred_key, got {required_pred_keys}."
            )

        super().__init__(
            name=name,
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
        context: Mapping[str, Any] | None = None,
    ) -> LossOut:
        ...

    @final
    def __call__(
        self,
        predictions: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> LossOut:
        skipped = self._skip_or_raise(
            predictions=predictions,
            targets=None,
            context=context,
            zero_anchor_pred_keys=self._required_pred_keys,  # required-first anchoring
        )
        if skipped is not None:
            return skipped
        out = self.loss(predictions, context=context)
        return self._postprocess(out)


class ContextualObjective(_BaseObjective):
    """
    Base class for contextual objectives.

    Contextual objectives compute loss primarily from:
    + context (required)
    Optionally they may use predictions and/or targets.
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
            raise ValueError(
                f"ContextualObjective {name} requires at least one required_context_key, got {required_context_keys}."
            )

        super().__init__(
            name=name,
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
        skipped = self._skip_or_raise(
            predictions=predictions,
            targets=targets,
            context=context,
            zero_anchor_pred_keys=self._required_pred_keys,  # required-first anchoring
        )
        if skipped is not None:
            return skipped
        out = self.loss(context, predictions=predictions, targets=targets)
        return self._postprocess(out)

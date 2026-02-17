from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

try:
    from typing import final  # Python 3.11+
except ImportError:
    from typing_extensions import final

import torch
from torch import Tensor


# A selector score is the scalar used for model selection / pruning / early stopping.
# It may be a single metric key or a weighted mixture of metric keys.
SelectorSpec = Union[str, Sequence[Tuple[str, float]]]


@dataclass
class EvalOut:
    """
    Output of an evaluator computation.

    + selector (float|int|Tensor): Scalar selection score.
      - If Tensor: must be 0-dim or single-element; will be reshaped to ().
    + metrics (Dict[str, Any]): All computed metrics and artifacts.
      - Values may be tensors, numpy arrays, nested dicts, or python scalars.
      - Scalars intended for logging/selection SHOULD be python floats/ints or scalar Tensors.
    """
    selector: float | int | Tensor
    metrics: Dict[str, Any] = field(default_factory=dict)


class _BaseEvaluator(ABC):
    """
    Base class for stateful evaluators that produce multiple metrics.

    Evaluators declare required keys from:
    + predictions (Mapping[str, Tensor|None])
    + targets     (Mapping[str, Tensor|None])
    + context     (Mapping[str, Any])

    Required key semantics:
    + A required key is considered missing if:
        - the key does not exist, OR
        - the key exists but its value is None.

    Lifecycle:
    + reset(): clear internal state before a new evaluation run (typically per epoch).
    + update(...): ingest one batch and update internal state (no return).
    + compute(): produce the aggregated EvalOut from the current state.

    Selection score (selector):
    + The evaluator computes a scalar "selector" used for selection/pruning.
    + The selector is defined by `selector` (SelectorSpec):
        - str: the name of a scalar metric key in the produced metrics dict.
        - Sequence[(key, weight)]: weighted mixture of scalar metric keys.
    + Missing / NaN terms are ignored and weights are renormalized over valid terms.

    Required / optional behavior:
    + required=True  -> missing required keys raise (full report).
    + required=False -> missing required keys skip:
        - __call__ returns a selector NaN and metrics describing missing keys,
        - update performs no-op,
        - compute remains unaffected by skipped calls.

    Scalar semantics:
    + EvalOut.selector MUST be scalar:
      - float/int, or
      - 0-dim Tensor (single-element tensors are reshaped to 0-dim).

    Notes:
    + This base does NOT enforce that all entries in `metrics` are scalars.
      Metrics may include arrays (e.g., confusion matrix), curves, or nested dicts.
    """

    def __init__(
        self,
        *,
        name: str,
        selector: SelectorSpec,
        required: bool = True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
        required_context_keys: tuple[str, ...] | list[str] = (),
    ):
        if not isinstance(name, str):
            raise TypeError(f"Evaluator name must be a string, got type {type(name)}.")
        if not name:
            raise ValueError(f"Evaluator name must be a non-empty string, got {name}.")
        if not isinstance(required, bool):
            raise TypeError(f"Evaluator {name} required flag must be a boolean, got type {type(required)}.")

        for nm, keys in (
            ("required_pred_keys", required_pred_keys),
            ("required_target_keys", required_target_keys),
            ("required_context_keys", required_context_keys),
        ):
            if not isinstance(keys, (tuple, list)):
                raise TypeError(f"Evaluator {name} {nm} must be a list or tuple of strings, got type {type(keys)}.")
            if any((not isinstance(k, str)) for k in keys):
                raise TypeError(f"Evaluator {name} {nm} must contain only strings, got {keys!r}.")
            if len(set(keys)) != len(keys):
                raise ValueError(f"Evaluator {name} {nm} contains duplicates: {keys!r}.")

        self._name = name
        self._required = required
        self._required_pred_keys = tuple(required_pred_keys)
        self._required_target_keys = tuple(required_target_keys)
        self._required_context_keys = tuple(required_context_keys)

        self._selector_spec: SelectorSpec = self._validate_selector_spec(selector, evaluator_name=name)

    @property
    def name(self) -> str:
        return self._name

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

    @property
    def selector_spec(self) -> SelectorSpec:
        return self._selector_spec

    def requirements(self) -> Dict[str, tuple[str, ...]]:
        return {
            "pred": self._required_pred_keys,
            "target": self._required_target_keys,
            "context": self._required_context_keys,
        }

    # ---- internal helpers ----

    @staticmethod
    def _validate_tensor_mapping(name: str, m: Mapping[str, Any] | None) -> None:
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
    def _first_tensor_in_mapping(m: Mapping[str, Any] | None) -> Tensor | None:
        if m is None:
            return None
        for v in m.values():
            if isinstance(v, Tensor):
                return v
        return None

    def _fallback_ref_tensor(
        self,
        *,
        predictions: Mapping[str, Any] | None,
        targets: Mapping[str, Any] | None,
        context: Mapping[str, Any] | None,
    ) -> Tensor | None:
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

    def _nan_selector_value(
        self,
        *,
        predictions: Mapping[str, Any] | None,
        targets: Mapping[str, Any] | None,
        context: Mapping[str, Any] | None,
    ) -> Tensor:
        ref = self._fallback_ref_tensor(predictions=predictions, targets=targets, context=context)
        if ref is None:
            return torch.tensor(float("nan"), dtype=torch.float32)
        return torch.tensor(float("nan"), device=ref.device, dtype=torch.float32)

    def _missing_info(
        self,
        *,
        predictions: Mapping[str, Tensor | None] | None,
        targets: Mapping[str, Tensor | None] | None,
        context: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        missing_pred = self._missing_keys(predictions, self._required_pred_keys)
        missing_target = self._missing_keys(targets, self._required_target_keys)
        missing_context = self._missing_keys(context, self._required_context_keys)

        info: Dict[str, Any] = {}
        if missing_pred:
            info["missing_pred_keys"] = missing_pred
        if missing_target:
            info["missing_target_keys"] = missing_target
        if missing_context:
            info["missing_context_keys"] = missing_context
        return info

    def _validate_and_maybe_skip(
        self,
        *,
        predictions: Mapping[str, Tensor | None] | None,
        targets: Mapping[str, Tensor | None] | None,
        context: Mapping[str, Any] | None,
    ) -> EvalOut | None:
        """
        Validate inputs and determine whether this update should be skipped.

        Returns:
        - None if requirements are satisfied (proceed with update).
        - EvalOut if skipped (optional evaluator missing keys).
        Raises:
        - KeyError for required evaluator missing keys.
        """
        self._validate_tensor_mapping("predictions", predictions)
        self._validate_tensor_mapping("targets", targets)
        if context is not None and not isinstance(context, Mapping):
            raise TypeError(f"context must be a Mapping[str, Any] or None, got {type(context)}.")

        info = self._missing_info(predictions=predictions, targets=targets, context=context)
        if not info:
            return None

        if self._required:
            parts: list[str] = []
            if "missing_pred_keys" in info:
                parts.append(f"prediction keys {info['missing_pred_keys']}")
            if "missing_target_keys" in info:
                parts.append(f"target keys {info['missing_target_keys']}")
            if "missing_context_keys" in info:
                parts.append(f"context keys {info['missing_context_keys']}")
            raise KeyError(f"Missing required {', '.join(parts)} for evaluator '{self._name}'.")

        # optional: skipped update => NaN selector + metrics describing missing keys
        v = self._nan_selector_value(predictions=predictions, targets=targets, context=context)
        return EvalOut(selector=v, metrics=info)

    @staticmethod
    def _coerce_scalar_value(v: float | int | Tensor, *, name: str) -> float | Tensor:
        if isinstance(v, (float, int)):
            return float(v)

        if isinstance(v, Tensor):
            if v.numel() == 0:
                raise ValueError(f"Evaluator '{name}' returned an empty tensor as scalar value.")
            if v.ndim == 0:
                return v
            if v.numel() == 1:
                return v.reshape(())
            raise ValueError(f"Evaluator '{name}' must return a scalar; got tensor with shape {tuple(v.shape)}.")

        raise TypeError(f"Evaluator '{name}' must return float/int or Tensor, got {type(v)}.")

    @staticmethod
    def _as_float_if_possible(v: Any) -> float | None:
        """
        Best-effort conversion of a scalar-like value to float for selector computation.
        Returns None if conversion is not possible or value is non-scalar.
        """
        if isinstance(v, (float, int)):
            fv = float(v)
            if not (fv != fv):  # not NaN
                return fv
            return float("nan")
        if isinstance(v, Tensor):
            if v.numel() != 1:
                return None
            fv = float(v.detach().cpu().reshape(()).item())
            return fv
        return None

    @staticmethod
    def _is_finite_number(x: float) -> bool:
        return (x == x) and (x != float("inf")) and (x != float("-inf"))

    @classmethod
    def _validate_selector_spec(cls, spec: SelectorSpec, *, evaluator_name: str) -> SelectorSpec:
        if isinstance(spec, str):
            if not spec:
                raise ValueError(f"Evaluator {evaluator_name}: selector key must be non-empty.")
            return spec

        if not isinstance(spec, (list, tuple)):
            raise TypeError(
                f"Evaluator {evaluator_name}: selector must be a str or a sequence of (key, weight), got {type(spec)}."
            )

        if not spec:
            raise ValueError(f"Evaluator {evaluator_name}: selector weighted spec must be non-empty.")

        out: list[tuple[str, float]] = []
        for i, item in enumerate(spec):
            if not (isinstance(item, (list, tuple)) and len(item) == 2):
                raise TypeError(
                    f"Evaluator {evaluator_name}: selector[{i}] must be a (key, weight) pair, got {item!r}."
                )
            k, w = item
            if not isinstance(k, str) or not k:
                raise TypeError(f"Evaluator {evaluator_name}: selector[{i}][0] must be a non-empty str.")
            w = float(w)
            if not cls._is_finite_number(w) or w < 0.0:
                raise ValueError(
                    f"Evaluator {evaluator_name}: selector[{i}] weight must be finite and >= 0, got {w}."
                )
            out.append((k, w))

        return tuple(out)

    def _compute_selector_from_metrics(self, metrics: Mapping[str, Any]) -> float | Tensor:
        """
        Compute the selector scalar from the produced metrics dict.

        Spec:
        - str: fetch metrics[key] and require scalar-like.
        - weighted: compute weighted sum over scalar-like terms; ignore NaN/missing terms
          and renormalize weights over valid terms.

        Returns float if possible, otherwise a scalar Tensor (rare; only if metric was Tensor).
        """
        spec = self._selector_spec

        if isinstance(spec, str):
            if spec not in metrics:
                return float("nan")
            v = metrics[spec]
            sv = self._as_float_if_possible(v)
            return float("nan") if sv is None else sv

        # weighted mixture
        terms: list[tuple[float, float]] = []
        for k, w in spec:
            if w <= 0.0:
                continue
            if k not in metrics:
                continue
            sv = self._as_float_if_possible(metrics[k])
            if sv is None:
                continue
            if not self._is_finite_number(sv):
                # allow NaN term to be skipped
                continue
            terms.append((w, sv))

        if not terms:
            return float("nan")

        wsum = sum(w for w, _ in terms)
        if wsum <= 0.0:
            return float("nan")

        return sum((w / wsum) * v for w, v in terms)

    def _postprocess(self, out: EvalOut) -> EvalOut:
        if not isinstance(out, EvalOut):
            raise TypeError(f"Evaluator '{self._name}' must return EvalOut, got {type(out)}.")
        out.selector = self._coerce_scalar_value(out.selector, name=self._name)
        if not isinstance(out.metrics, dict):
            raise TypeError(f"Evaluator '{self._name}' must return metrics as dict, got {type(out.metrics)}.")
        return out

    # ---- stateful lifecycle ----

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state before a new evaluation run."""
        ...

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Ingest one batch and update internal state."""
        ...

    @abstractmethod
    def compute_metrics(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Compute and return all aggregated metrics/artifacts from the current internal state.

        This must return a dictionary. It may include:
        - scalar metrics (float/int/scalar Tensor)
        - arrays / nested dicts (artifacts)
        """
        ...

    def compute(self, *args: Any, **kwargs: Any) -> EvalOut:
        """
        Compute the aggregated EvalOut from the current internal state.
        """
        metrics = self.compute_metrics(*args, **kwargs)
        if not isinstance(metrics, dict):
            raise TypeError(f"Evaluator '{self._name}' compute_metrics() must return dict, got {type(metrics)}.")
        selector = self._compute_selector_from_metrics(metrics)
        return self._postprocess(EvalOut(selector=selector, metrics=metrics))

    # ---- repr/str ----

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"required={self._required}, "
            f"selector_spec={self._selector_spec!r}, "
            f"required_pred_keys={self._required_pred_keys}, "
            f"required_target_keys={self._required_target_keys}, "
            f"required_context_keys={self._required_context_keys}"
            f")"
        )

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name})"


class RelationalEvaluator(_BaseEvaluator):
    """
    Base class for relational evaluators.

    Relational evaluators update internal state by comparing:
    + predictions against targets
    + optional context (e.g., masks, weights, metadata)

    Calling the evaluator performs a validated update and returns:
    - `{name}`: selector scalar
    - `{name}/{k}`: each entry from the aggregated metrics dict (best-effort detaching tensors)
    """

    def __init__(
        self,
        *,
        name: str,
        selector: SelectorSpec,
        required: bool = True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
        required_context_keys: tuple[str, ...] | list[str] = (),
    ):
        if not required_pred_keys:
            raise ValueError(
                f"RelationalEvaluator {name} requires at least one required_pred_key, got {required_pred_keys}."
            )
        if not required_target_keys:
            raise ValueError(
                f"RelationalEvaluator {name} requires at least one required_target_key, got {required_target_keys}."
            )

        super().__init__(
            name=name,
            selector=selector,
            required=required,
            required_pred_keys=required_pred_keys,
            required_target_keys=required_target_keys,
            required_context_keys=required_context_keys,
        )

    @abstractmethod
    def update(
        self,
        predictions: Mapping[str, Tensor | None],
        targets: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> None:
        ...

    @final
    def __call__(
        self,
        predictions: Mapping[str, Tensor | None],
        targets: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        skipped = self._validate_and_maybe_skip(predictions=predictions, targets=targets, context=context)
        if skipped is None:
            self.update(predictions, targets, context=context)
            out = self.compute()
        else:
            out = skipped

        out = self._postprocess(out)

        flat: Dict[str, Any] = {self.name: out.selector}
        for k, v in out.metrics.items():
            # don't detach numpy / python objects; detach tensors for safety
            flat[f"{self.name}/{k}"] = v.detach() if isinstance(v, Tensor) else v
        return flat


class IntrinsicEvaluator(_BaseEvaluator):
    """
    Base class for intrinsic evaluators.

    Intrinsic evaluators update internal state from:
    + predictions (required)
    + optional context

    Calling the evaluator performs a validated update and returns:
    - `{name}`: selector scalar
    - `{name}/{k}`: each aggregated metric/artifact
    """

    def __init__(
        self,
        *,
        name: str,
        selector: SelectorSpec,
        required: bool = True,
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_context_keys: tuple[str, ...] | list[str] = (),
    ):
        if not required_pred_keys:
            raise ValueError(
                f"IntrinsicEvaluator {name} requires at least one required_pred_key, got {required_pred_keys}."
            )

        super().__init__(
            name=name,
            selector=selector,
            required=required,
            required_pred_keys=required_pred_keys,
            required_target_keys=(),
            required_context_keys=required_context_keys,
        )

    @abstractmethod
    def update(
        self,
        predictions: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> None:
        ...

    @final
    def __call__(
        self,
        predictions: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        skipped = self._validate_and_maybe_skip(predictions=predictions, targets=None, context=context)
        if skipped is None:
            self.update(predictions, context=context)
            out = self.compute()
        else:
            out = skipped

        out = self._postprocess(out)

        flat: Dict[str, Any] = {self.name: out.selector}
        for k, v in out.metrics.items():
            flat[f"{self.name}/{k}"] = v.detach() if isinstance(v, Tensor) else v
        return flat


class ContextualEvaluator(_BaseEvaluator):
    """
    Base class for contextual evaluators.

    Contextual evaluators update internal state primarily from:
    + context (required)

    Optionally they may use predictions and/or targets.

    Calling the evaluator performs a validated update and returns:
    - `{name}`: selector scalar
    - `{name}/{k}`: each aggregated metric/artifact
    """

    def __init__(
        self,
        *,
        name: str,
        selector: SelectorSpec,
        required: bool = True,
        required_context_keys: tuple[str, ...] | list[str] = (),
        required_pred_keys: tuple[str, ...] | list[str] = (),
        required_target_keys: tuple[str, ...] | list[str] = (),
    ):
        if not required_context_keys:
            raise ValueError(
                f"ContextualEvaluator {name} requires at least one required_context_key, got {required_context_keys}."
            )

        super().__init__(
            name=name,
            selector=selector,
            required=required,
            required_pred_keys=required_pred_keys,
            required_target_keys=required_target_keys,
            required_context_keys=required_context_keys,
        )

    @abstractmethod
    def update(
        self,
        context: Mapping[str, Any],
        predictions: Mapping[str, Tensor | None] | None = None,
        targets: Mapping[str, Tensor | None] | None = None,
    ) -> None:
        ...

    @final
    def __call__(
        self,
        context: Mapping[str, Any],
        predictions: Mapping[str, Tensor | None] | None = None,
        targets: Mapping[str, Tensor | None] | None = None,
    ) -> Dict[str, Any]:
        skipped = self._validate_and_maybe_skip(predictions=predictions, targets=targets, context=context)
        if skipped is None:
            self.update(context, predictions=predictions, targets=targets)
            out = self.compute()
        else:
            out = skipped

        out = self._postprocess(out)

        flat: Dict[str, Any] = {self.name: out.selector}
        for k, v in out.metrics.items():
            flat[f"{self.name}/{k}"] = v.detach() if isinstance(v, Tensor) else v
        return flat

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Sequence

import numbers
from typing import Optional

from torch import Tensor, nn

MetricDirection = Literal["minimize", "maximize"]


class Evaluator(ABC):
    """
    Base class for Evaluators.

    Evaluators compute metrics from model outputs and batch payload.

    Contract:
    - `forward(inputs=...)` receives the same nested dict used by Objectives.
    - Derived classes define required tensor paths using "/" namespace paths.
    - Metrics are returned as dict[str, Tensor], each Tensor must be scalar.
    - One metric must be designated as the `primary_metric` for optimization.
    """

    def __init__(
        self,
        *,
        name: str,
        primary_metric: str,
        direction: Literal["maximize", "minimize"] = "maximize",
        weight: float = 1.0,
    ):
        super().__init__()

        if not isinstance(name, str) or not name:
            raise ValueError("Evaluator `name` must be a non-empty string.")

        if not isinstance(primary_metric, str) or not primary_metric:
            raise ValueError("Evaluator `primary_metric` must be a non-empty string.")

        if direction not in ("maximize", "minimize"):
            raise ValueError("direction must be 'maximize' or 'minimize'.")

        if not isinstance(weight, (int, float)):
            raise TypeError(f"Evaluator `weight` must be a number, got {type(weight).__name__}.")
        if float(weight) < 0.0:
            raise ValueError(f"Evaluator `weight` must be non-negative, got {weight}.")

        self._name = name
        self._primary_metric = primary_metric
        self._direction = direction
        self._weight = float(weight)

    # properties ---

    @property
    def name(self) -> str:
        return self._name

    @property
    def primary_metric(self) -> str:
        return self._primary_metric

    @property
    def direction(self) -> Literal["maximize", "minimize"]:
        return self._direction

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def required_keys(self) -> tuple[str, ...]:
        raise NotImplementedError(
            f"{self.__class__.__name__} must define `required_keys`."
        )
    
    @property
    def optional_keys(self) -> tuple[str, ...]:
        """Optional keys that may be None."""
        return tuple()

    # helpers ---

    @staticmethod
    def resolve(inputs: dict[str, Any], key: str, strict=True) -> Tensor | None:
        """Strict resolver requires leaf Tensor. Non strict allows None values (useful for optional metrics)."""
        current: Any = inputs
        parts = [p for p in key.split("/") if p]

        for i, part in enumerate(parts):
            path = "/".join(parts[:i]) or "<root>"

            if not isinstance(current, dict):
                raise TypeError(f"Expected dict at path {path}, got {type(current).__name__}.")

            if part not in current:
                raise KeyError(
                    f"Key {part!r} not found at path {path}. Available keys: {list(current.keys())}."
                )

            current = current[part]
            
        if not strict and current is None:
            return None

        if not isinstance(current, Tensor):
            raise TypeError(
                f"Resolved value for key {key!r} must be Tensor, got {type(current).__name__}."
            )

        return current

    def _missing_keys(self, inputs: dict[str, Any]) -> list[str]:
        missing: list[str] = []
        required = set(self.required_keys)
        optional = set(self.optional_keys)

        # Optional keys are allowed to be None, but they still must exist in the payload
        # (your stated contract: stable keys with None values).
        to_check = required | optional  # set union

        for key in to_check:
            try:
                _ = self.resolve(inputs, key, strict=(key not in optional))
            except (KeyError, TypeError):
                # Only "required" keys can make the evaluator fail
                if key in required:
                    missing.append(key)

        return missing

    # metric implementation ---

    @abstractmethod
    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Implement metric computation here.
        """
        raise NotImplementedError

    # interface ---

    def __call__(self, *, inputs: dict[str, Any]) -> dict[str, Any]:

        if not isinstance(inputs, dict):
            raise TypeError(f"Evaluator inputs must be dict[str, Any], got {type(inputs).__name__}.")

        missing = self._missing_keys(inputs)
        if missing:
            raise KeyError(
                f"Evaluator '{self.name}' missing required keys: {missing}. "
                f"Top-level keys: {list(inputs.keys())}."
            )

        metrics = self.metrics(inputs=inputs)

        if not isinstance(metrics, dict):
            raise TypeError("Evaluator must return dict[str, Any].")

        for k, v in metrics.items():
            if not isinstance(k, str):
                raise TypeError("Metric names must be strings.")

        if self.primary_metric not in metrics:
            raise KeyError(
                f"Primary metric '{self.primary_metric}' not found in returned metrics {list(metrics.keys())}."
            )

        return metrics

    # OPTIONAL but very useful for Trainer/CompositeEvaluator:
    def primary_value(self, *, metrics: dict[str, Tensor | None]) -> Optional[float]:
        v = metrics[self.primary_metric]
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, numbers.Number):
            raise TypeError(
                f"Primary metric '{self.primary_metric}' must be a number or None, got {type(v).__name__}."
            )
        return float(v)
    

class CompositeEvaluator(Evaluator):
    """
    Minimal CompositeEvaluator:

    - Runs each child Evaluator on the same `inputs`.
    - Aggregates metrics under namespaced keys: "{evaluator.name}/{metric_name}".
    - Computes composite primary as weighted sum of child primaries:
          __primary__ = Σ_i (w_i * primary_i)
      where primary_i is the child's primary metric value.

    Notes:
    - No flattening of nested composites.
    - No direction validation.
    - Child evaluator names must be unique (otherwise namespacing collides).
    """

    def __init__(
        self,
        evaluators: Sequence[Evaluator],
        *,
        name: str = "composite",
        primary_metric: str = "__primary__",
        direction: MetricDirection = "maximize",
    ) -> None:
        if not isinstance(evaluators, (list, tuple)) or len(evaluators) == 0:
            raise ValueError("CompositeEvaluator requires a non-empty list/tuple of evaluators.")
        for ev in evaluators:
            if not isinstance(ev, Evaluator):
                raise TypeError(f"All items must be Evaluator, got {type(ev).__name__}.")

        # ensure unique names to avoid collisions
        names = [ev.name for ev in evaluators]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Child evaluator names must be unique, duplicates: {dupes}.")

        super().__init__(
            name=name,
            primary_metric=primary_metric,  # composite primary key
            direction=direction,
            weight=1.0,  # composite itself typically not reweighted
        )

        self.evaluators: list[Evaluator] = list(evaluators)

    @property
    def required_keys(self) -> tuple[str, ...]:
        """Union of required keys across child evaluators."""
        keys = set()
        for ev in self.evaluators:
            keys.update(ev.required_keys)
        return tuple(sorted(keys))
    
    @property
    def optional_keys(self) -> tuple[str, ...]:
        """Union of optional keys across child evaluators."""
        keys = set()
        for ev in self.evaluators:
            keys.update(ev.optional_keys)
        return tuple(sorted(keys))

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        primary: Any | None = None

        for ev in self.evaluators:
            m = ev(inputs=inputs)  # strict checks happen inside child evaluator

            # namespace all metrics
            prefix = ev.name
            for k, v in m.items():
                nk = f"{prefix}/{k}"
                if nk in out:
                    raise KeyError(f"Metric key collision after namespacing: {nk!r}")
                out[nk] = v

            # pull child's primary (from the child's returned metrics)
            child_primary_key = f"{prefix}/{ev.primary_metric}"
            if child_primary_key not in out:
                # this should never happen if child evaluator is correct
                raise KeyError(
                    f"Child evaluator '{ev.name}' did not return its primary metric "
                    f"'{ev.primary_metric}'. Returned: {list(m.keys())}"
                )

            p = out[child_primary_key]
            if p is None:
                continue  # skip None values in composite primary calculation

            if isinstance(p, bool) or not isinstance(p, numbers.Number):
                raise TypeError(
                    f"Child evaluator '{ev.name}' primary metric '{ev.primary_metric}' must be a number or None, got {type(p).__name__}."
                )

            w = float(ev.weight)
            term = p * w

            # Align direction with composite evaluator
            if ev.direction != self.direction:
                term = -term    # flip sign if child's direction differs from composite's

            primary = term if primary is None else (primary + term)

        if self.primary_metric in out:
            raise KeyError(
                f"Composite primary metric name {self.primary_metric!r} collides with an existing metric key. "
                f"Choose a different primary_metric."
            )

        out[self.primary_metric] = primary
        return out
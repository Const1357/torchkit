from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Sequence

from torch import Tensor

from torchkit.evaluate._evaluator import Evaluator

MetricDirection = Literal["minimize", "maximize"]


class SelectorEvaluator(Evaluator, ABC):
    def __init__(
        self,
        *,
        name: str,
        direction: MetricDirection,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name)

        if direction not in ("minimize", "maximize"):
            raise ValueError(
                f"direction must be 'minimize' or 'maximize', got {direction!r}."
            )

        if not isinstance(weight, (int, float)):
            raise TypeError(f"weight must be numeric, got {type(weight).__name__}.")

        self._direction = direction
        self._weight = float(weight)

    @property
    def direction(self) -> MetricDirection:
        return self._direction

    @property
    def weight(self) -> float:
        return self._weight

    @abstractmethod
    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        raise NotImplementedError

    def selector_spec(self) -> dict[str, Any]:
        return {
            "type": "selector",
            "name": self.name,
            "direction": self.direction,
            "weight": self.weight,
        }

    def compute(self, *, inputs: dict[str, Any]) -> tuple[Tensor, dict[str, dict[str, Any]]]:
        self._validate_inputs(inputs=inputs, kind="SelectorEvaluator")

        primary = self.primary_metric(inputs=inputs)

        if not isinstance(primary, Tensor):
            raise TypeError("SelectorEvaluator must return a Tensor.")

        if primary.numel() != 1:
            raise ValueError(
                f"SelectorEvaluator must return a scalar Tensor, got shape {tuple(primary.shape)}."
            )

        raw = float(primary.detach().cpu().item())
        signed = raw if self.direction == "maximize" else -raw
        weighted = signed * self.weight

        components = {
            self.name: {
                "raw": raw,
                "direction": self.direction,
                "weight": self.weight,
                "signed": signed,
                "weighted": weighted,
            }
        }
        return primary, components

    def selector_components(self, *, inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
        _, components = self.compute(inputs=inputs)
        return components

    def __call__(self, *, inputs: dict[str, Any]) -> Tensor:
        primary, _ = self.compute(inputs=inputs)
        return primary


class CompositeSelectorEvaluator(SelectorEvaluator):
    def __init__(
        self,
        evaluators: Sequence[SelectorEvaluator],
        *,
        name: str = "composite_selector",
    ) -> None:
        if not isinstance(evaluators, (list, tuple)) or len(evaluators) == 0:
            raise ValueError(
                "CompositeSelectorEvaluator requires a non-empty list/tuple of selector evaluators."
            )

        for ev in evaluators:
            if not isinstance(ev, SelectorEvaluator):
                raise TypeError(
                    f"All items must be SelectorEvaluator, got {type(ev).__name__}."
                )

        names = [ev.name for ev in evaluators]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Child evaluator names must be unique, duplicates: {dupes}.")

        super().__init__(name=name, direction="maximize", weight=1.0)
        self.evaluators: list[SelectorEvaluator] = list(evaluators)

    @property
    def required_keys(self) -> tuple[str, ...]:
        keys = set()
        for ev in self.evaluators:
            keys.update(ev.required_keys)
        return tuple(sorted(keys))

    @property
    def optional_keys(self) -> tuple[str, ...]:
        keys = set()
        for ev in self.evaluators:
            keys.update(ev.optional_keys)
        return tuple(sorted(keys))

    def selector_spec(self) -> dict[str, Any]:
        return {
            "type": "composite_selector",
            "name": self.name,
            "direction": self.direction,
            "weight": self.weight,
            "children": [ev.selector_spec() for ev in self.evaluators],
        }

    def compute(self, *, inputs: dict[str, Any]) -> tuple[Tensor, dict[str, dict[str, Any]]]:
        self._validate_inputs(inputs=inputs, kind="SelectorEvaluator")

        total: Tensor | None = None
        components: dict[str, dict[str, Any]] = {}

        for ev in self.evaluators:
            value, child_components = ev.compute(inputs=inputs)

            signed = value if ev.direction == "maximize" else -value
            contribution = signed * ev.weight

            total = contribution if total is None else total + contribution

            for key, payload in child_components.items():
                components[f"{self.name}/{key}"] = payload

        if total is None:
            raise RuntimeError("CompositeSelectorEvaluator has no child evaluators.")

        total_raw = float(total.detach().cpu().item())
        components[self.name] = {
            "raw": total_raw,
            "direction": "maximize",
            "weight": 1.0,
            "signed": total_raw,
            "weighted": total_raw,
        }

        return total, components

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        total, _ = self.compute(inputs=inputs)
        return total
from __future__ import annotations

from abc import ABC, abstractmethod

from torchkit.evaluate._evaluator import Evaluator
from typing import Any, Sequence


class ReportEvaluator(Evaluator, ABC):
    """
    Base class for report evaluators.

    Contract:
    - receives nested payload via `inputs`
    - returns dict[str, Any]
    - may return multiple metrics
    - no primary metric semantics here
    """

    @abstractmethod
    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def __call__(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        self._validate_inputs(inputs=inputs, kind="ReportEvaluator")

        metrics = self.metrics(inputs=inputs)

        if not isinstance(metrics, dict):
            raise TypeError("ReportEvaluator must return dict[str, Any].")

        for k in metrics.keys():
            if not isinstance(k, str):
                raise TypeError("Metric names must be strings.")

        return metrics


class CompositeReportEvaluator(ReportEvaluator):
    """
    Composite report evaluator.

    - Runs each child ReportEvaluator on the same inputs.
    - Namespaces child metrics as: "{child.name}/{metric_name}".
    - Child evaluator names must be unique.
    """

    def __init__(
        self,
        evaluators: Sequence[ReportEvaluator],
        *,
        name: str = "composite_report",
    ) -> None:
        if not isinstance(evaluators, (list, tuple)) or len(evaluators) == 0:
            raise ValueError(
                "CompositeReportEvaluator requires a non-empty list/tuple of report evaluators."
            )

        for ev in evaluators:
            if not isinstance(ev, ReportEvaluator):
                raise TypeError(
                    f"All items must be ReportEvaluator, got {type(ev).__name__}."
                )

        names = [ev.name for ev in evaluators]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Child evaluator names must be unique, duplicates: {dupes}.")

        super().__init__(name=name)
        self.evaluators: list[ReportEvaluator] = list(evaluators)

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

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}

        for ev in self.evaluators:
            child_metrics = ev(inputs=inputs)

            for metric_name, value in child_metrics.items():
                namespaced = f"{ev.name}/{metric_name}"
                if namespaced in out:
                    raise KeyError(
                        f"Metric key collision after namespacing: {namespaced!r}"
                    )
                out[namespaced] = value

        return out
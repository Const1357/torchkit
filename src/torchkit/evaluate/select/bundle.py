from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torchkit.evaluate.select._selector_evaluator import SelectorEvaluator


@dataclass(frozen=True)
class BundleSelectorEvaluator:
    batch_evaluator: Optional[SelectorEvaluator] = None
    dataset_evaluator: Optional[SelectorEvaluator] = None

    def __post_init__(self) -> None:
        if self.batch_evaluator is None and self.dataset_evaluator is None:
            raise ValueError(
                "BundleSelectorEvaluator requires at least one of "
                "`batch_evaluator` or `dataset_evaluator`."
            )
        if self.batch_evaluator is not None and not isinstance(self.batch_evaluator, SelectorEvaluator):
            raise TypeError(
                f"`batch_evaluator` must be a SelectorEvaluator, got {type(self.batch_evaluator).__name__}."
            )
        if self.dataset_evaluator is not None and not isinstance(self.dataset_evaluator, SelectorEvaluator):
            raise TypeError(
                f"`dataset_evaluator` must be a SelectorEvaluator, got {type(self.dataset_evaluator).__name__}."
            )

    @property
    def has_batch_evaluator(self) -> bool:
        return self.batch_evaluator is not None

    @property
    def has_dataset_evaluator(self) -> bool:
        return self.dataset_evaluator is not None

    @property
    def batch_required_keys(self) -> tuple[str, ...]:
        if self.batch_evaluator is None:
            return tuple()
        return self.batch_evaluator.required_keys

    @property
    def batch_optional_keys(self) -> tuple[str, ...]:
        if self.batch_evaluator is None:
            return tuple()
        return self.batch_evaluator.optional_keys

    @property
    def dataset_required_keys(self) -> tuple[str, ...]:
        if self.dataset_evaluator is None:
            return tuple()
        return self.dataset_evaluator.required_keys

    @property
    def dataset_optional_keys(self) -> tuple[str, ...]:
        if self.dataset_evaluator is None:
            return tuple()
        return self.dataset_evaluator.optional_keys
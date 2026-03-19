from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torchkit.evaluate.report._report_evaluator import ReportEvaluator

@dataclass(frozen=True)
class BundleReportEvaluator:
    batch_evaluator: Optional[ReportEvaluator] = None
    dataset_evaluator: Optional[ReportEvaluator] = None

    def __post_init__(self) -> None:
        if self.batch_evaluator is None and self.dataset_evaluator is None:
            raise ValueError(
                "BundleReportEvaluator requires at least one of "
                "`batch_evaluator` or `dataset_evaluator`."
            )
        if self.batch_evaluator is not None and not isinstance(self.batch_evaluator, ReportEvaluator):
            raise TypeError(
                f"`batch_evaluator` must be a ReportEvaluator, got {type(self.batch_evaluator).__name__}."
            )
        if self.dataset_evaluator is not None and not isinstance(self.dataset_evaluator, ReportEvaluator):
            raise TypeError(
                f"`dataset_evaluator` must be a ReportEvaluator, got {type(self.dataset_evaluator).__name__}."
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



from __future__ import annotations

from typing import Any, Callable, Optional, Literal

import copy
import os

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from torchkit.data._dataset import TorchkitDataset
from torchkit.data.split import KFoldSplitter
from torchkit.models.Model.factory import TorchkitModelSpec
from torchkit.train.factory import TrainerSpec
from torchkit.train.trainer import Trainer


MetricDirection = Literal["maximize", "minimize"]


def _safe_take(values: Any, indices: list[int] | tuple[int, ...] | Any) -> Any:
    if values is None:
        return None

    if hasattr(values, "take"):
        try:
            return values.take(indices)
        except Exception:
            pass

    if hasattr(values, "iloc"):
        try:
            return values.iloc[list(indices)]
        except Exception:
            pass

    if torch.is_tensor(values):
        return values[list(indices)]

    return [values[i] for i in indices]


def _clone_tensor_dict(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in d.items()}


def _clone_state_dict_cpu(
    sd: Optional[dict[str, torch.Tensor]],
) -> Optional[dict[str, torch.Tensor]]:
    if sd is None:
        return None
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def _concat_tensor_dicts(dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    buckets: dict[str, list[torch.Tensor]] = {}
    for d in dicts:
        for k, v in d.items():
            buckets.setdefault(k, []).append(v.detach().cpu())

    merged: dict[str, torch.Tensor] = {}
    for k, vs in buckets.items():
        if len(vs) == 1:
            merged[k] = vs[0].clone()
        else:
            merged[k] = torch.cat(vs, dim=0)
    return merged


def _resolve_original_indices_for_subset(subset: Subset) -> list[int]:
    """
    Resolve nested Subset indices back to original dataset coordinates.
    """
    indices = list(subset.indices)
    base = subset.dataset

    while isinstance(base, Subset):
        parent_indices = list(base.indices)
        indices = [parent_indices[i] for i in indices]
        base = base.dataset

    return indices


class BaseCV:
    """
    Base class for all CV runners.

    Contains only generic CV infrastructure:
    - splitter construction / dispatch
    - dataloader construction
    - selection metric helpers
    - holdout evaluation
    - calibrator fitting from OOF tensors

    Search-specific logic belongs in BaseSearchCV.
    Optuna-specific logic belongs in OptunaSearchMixin.
    """

    def __init__(
        self,
        *,
        model_spec: TorchkitModelSpec,
        trainer_spec: TrainerSpec,
        outer_splitter_cls: type[KFoldSplitter],
        inner_splitter_cls: Optional[type[KFoldSplitter]] = None,
        dataloader_factory: Optional[Callable[[Dataset, bool], DataLoader]] = None,
        k_outer: int = 5,
        k_inner: Optional[int] = None,
        shuffle_outer: bool = False,
        shuffle_inner: bool = False,
        random_state: Optional[int] = None,
        calibrate: bool = True,
        final_model_dir: Optional[str] = None,
        keep_final_model_state_dict_cpu: bool = True,
    ):
        self.model_spec = copy.deepcopy(model_spec)
        self.trainer_spec = copy.deepcopy(trainer_spec)

        self.outer_splitter_cls = outer_splitter_cls
        self.inner_splitter_cls = inner_splitter_cls

        self.k_outer = int(k_outer)
        self.k_inner = int(k_inner) if k_inner is not None else None
        self.shuffle_outer = bool(shuffle_outer)
        self.shuffle_inner = bool(shuffle_inner)
        self.random_state = random_state

        self.calibrate = bool(calibrate)
        self.final_model_dir = final_model_dir
        self.keep_final_model_state_dict_cpu = bool(keep_final_model_state_dict_cpu)

        if self.final_model_dir is not None:
            os.makedirs(self.final_model_dir, exist_ok=True)

        if (self.final_model_dir is None) and (not self.keep_final_model_state_dict_cpu):
            raise ValueError(
                "Final models would be unrebuildable: both final_model_dir is None and "
                "keep_final_model_state_dict_cpu is False."
            )

        self.outer_splitter = outer_splitter_cls(
            n_splits=self.k_outer,
            shuffle=self.shuffle_outer,
            random_state=self.random_state,
        )

        self.inner_splitter = None
        if inner_splitter_cls is not None:
            if self.k_inner is None:
                raise ValueError("k_inner must be provided when inner_splitter_cls is provided.")

            self.inner_splitter = inner_splitter_cls(
                n_splits=self.k_inner,
                shuffle=self.shuffle_inner,
                random_state=self.random_state,
            )

        if dataloader_factory is None:
            self.dataloader_factory = lambda ds, shuffle: DataLoader(ds, batch_size=1, shuffle=shuffle)
        else:
            self.dataloader_factory = dataloader_factory

    def _split(
        self,
        splitter: KFoldSplitter,
        dataset: TorchkitDataset | Subset,
        y: Any,
        groups: Optional[Any] = None,
    ):
        if groups is None:
            return splitter.split(dataset, y)
        return splitter.split(dataset, y, groups)

    def _selection_metric_name(self) -> str:
        dataset_evaluator = getattr(self.trainer_spec, "dataset_evaluator", None)
        if dataset_evaluator is not None:
            return str(dataset_evaluator.primary_metric)
        return "val_loss"

    def _selection_metric_direction(self) -> MetricDirection:
        dataset_evaluator = getattr(self.trainer_spec, "dataset_evaluator", None)
        if dataset_evaluator is not None:
            return str(dataset_evaluator.direction)  # type: ignore[return-value]
        return "minimize"

    def _to_selection_score(self, raw_metric: float) -> float:
        direction = self._selection_metric_direction()
        if direction == "maximize":
            return float(raw_metric)
        if direction == "minimize":
            return -float(raw_metric)
        raise ValueError(f"Unsupported selection metric direction {direction!r}.")

    def _evaluate_holdout(
        self,
        trainer: Trainer,
        dataset_subset: Subset | Dataset,
    ) -> dict[str, Any]:
        loader = self.dataloader_factory(dataset_subset, False)
        state_backup = copy.deepcopy(trainer.state)
        try:
            metrics = trainer._validate_one_epoch(loader, epoch=0)
        finally:
            trainer.state = state_backup
        return metrics

    def _fit_calibrators_from_oof(
        self,
        model: Any,
        *,
        oof_logits: dict[str, torch.Tensor],
        oof_targets: dict[str, torch.Tensor],
    ) -> None:
        """
        Fit active calibrators from OOF tensors.

        This method is intentionally generic and does not depend on any
        concrete result dataclass.
        """
        if not self.calibrate:
            return

        prediction_heads = getattr(model, "prediction_heads", None)
        if prediction_heads is None:
            return

        for task, prediction_head in prediction_heads.items():
            if prediction_head is None or not getattr(prediction_head, "is_active", True):
                continue

            calibrator = getattr(prediction_head, "calibrator", None)
            if calibrator is None or not getattr(calibrator, "is_active", True):
                continue

            if task not in oof_logits or task not in oof_targets:
                raise ValueError(
                    f"Calibrator for task {task!r} is active, but OOF logits/targets are missing."
                )

            calibrator.fit(
                logits=oof_logits[task],
                targets=oof_targets[task],
            )
            calibrator.enable()

    @staticmethod
    def _assert_exact_oof_coverage(
        *,
        sample_indices: list[int],
        reference_indices: list[int],
        context: str,
    ) -> None:
        if not sample_indices:
            return

        if len(sample_indices) != len(set(sample_indices)):
            raise ValueError(
                f"{context} produced duplicated OOF sample indices. "
                "This indicates leakage or overlapping validation folds."
            )

        if sorted(sample_indices) != sorted(reference_indices):
            raise ValueError(
                f"{context} produced OOF sample indices that do not exactly cover the "
                "reference training pool. This indicates missing or leaked samples."
            )
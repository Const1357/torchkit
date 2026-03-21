from __future__ import annotations

from typing import Any, Callable, Optional, Literal

import copy
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from torchkit.data._dataset import TorchkitDataset
from torchkit.data.split import KFoldSplitter
from torchkit.evaluate import evaluate as evaluate_model
from torchkit.evaluate.report.bundle import BundleReportEvaluator
from torchkit.models.Model._model import TorchkitModel
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


def _module_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _resolve_original_indices_for_subset(subset: Subset) -> list[int]:
    indices = list(subset.indices)
    base = subset.dataset

    while isinstance(base, Subset):
        parent_indices = list(base.indices)
        indices = [parent_indices[i] for i in indices]
        base = base.dataset

    return indices


class BaseCV:
    @staticmethod
    def _coerce_model_spec(model_spec: TorchkitModelSpec | TorchkitModel) -> TorchkitModelSpec:
        if isinstance(model_spec, TorchkitModelSpec):
            return copy.deepcopy(model_spec)

        if isinstance(model_spec, TorchkitModel):
            spec = model_spec.to_spec()
            if not isinstance(spec, TorchkitModelSpec):
                raise TypeError(
                    f"{model_spec.__class__.__name__}.to_spec() must return TorchkitModelSpec."
                )
            return copy.deepcopy(spec)

        raise TypeError(
            "`model_spec` must be a TorchkitModelSpec or a live TorchkitModel instance, "
            f"got {type(model_spec).__name__}."
        )

    def __init__(
        self,
        *,
        model_spec: TorchkitModelSpec | TorchkitModel,
        trainer_spec: TrainerSpec,
        splitter_cls: type[KFoldSplitter],
        dataloader_factory: Optional[Callable[[Dataset, bool], DataLoader]] = None,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
        calibrate: bool = True,
        report_evaluator: Optional[BundleReportEvaluator] = None,
        final_model_dir: Optional[str] = None,
        keep_final_model_state_dict_cpu: bool = True,
    ):
        self.model_spec = self._coerce_model_spec(model_spec)
        self.trainer_spec = copy.deepcopy(trainer_spec)

        self.splitter_cls = splitter_cls
        self.n_splits = int(n_splits)
        self.shuffle = bool(shuffle)
        self.random_state = random_state

        self.calibrate = bool(calibrate)
        self.report_evaluator = copy.deepcopy(report_evaluator)
        self.final_model_dir = final_model_dir
        self.keep_final_model_state_dict_cpu = bool(keep_final_model_state_dict_cpu)

        if self.final_model_dir is not None:
            os.makedirs(self.final_model_dir, exist_ok=True)

        if (self.final_model_dir is None) and (not self.keep_final_model_state_dict_cpu):
            raise ValueError(
                "Final models would be unrebuildable: both final_model_dir is None and "
                "keep_final_model_state_dict_cpu is False."
            )

        self.splitter = splitter_cls(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
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
        selector_bundle = getattr(self.trainer_spec, "selector_evaluator", None)
        if selector_bundle is not None:
            parts: list[str] = []
            batch_selector = getattr(selector_bundle, "batch_evaluator", None)
            dataset_selector = getattr(selector_bundle, "dataset_evaluator", None)
            if batch_selector is not None:
                parts.append(f"batch:{batch_selector.name}")
            if dataset_selector is not None:
                parts.append(f"dataset:{dataset_selector.name}")
            return " + ".join(parts) if parts else "selector_primary"
        return "val_loss"
    
    def _selection_metric_spec(self) -> dict[str, Any]:
        selector_bundle = getattr(self.trainer_spec, "selector_evaluator", None)
        if selector_bundle is not None:
            batch_selector = getattr(selector_bundle, "batch_evaluator", None)
            dataset_selector = getattr(selector_bundle, "dataset_evaluator", None)
            return {
                "type": "selector_bundle",
                "direction": "maximize",
                "batch_evaluator": None if batch_selector is None else batch_selector.selector_spec(),
                "dataset_evaluator": None if dataset_selector is None else dataset_selector.selector_spec(),
            }

        return {
            "type": "val_loss_fallback",
            "name": "val_loss",
            "direction": "minimize",
        }

    def _selection_metric_direction(self) -> MetricDirection:
        selector_bundle = getattr(self.trainer_spec, "selector_evaluator", None)
        if selector_bundle is not None:
            return "maximize"
        return "minimize"

    def _to_selection_score(self, raw_metric: float) -> float:
        selector_bundle = getattr(self.trainer_spec, "selector_evaluator", None)
        if selector_bundle is not None:
            return float(raw_metric)

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

    def _evaluate_report(
        self,
        trainer: Trainer,
        dataset_subset: Subset | Dataset,
    ) -> Optional[dict[str, Any]]:
        if self.report_evaluator is None:
            return None

        return evaluate_model(
            trainer.model,
            dataset_subset,
            self.report_evaluator,
            device=trainer.device,
            backbone_kwargs=trainer.config.backbone_kwargs,
            head_kwargs=trainer.config.head_kwargs,
            use_amp=trainer.config.use_amp,
            dataloader_factory=lambda ds: self.dataloader_factory(ds, False),
        )

    def _fit_posthoc_modules_from_oof(
        self,
        model: Any,
        *,
        oof_logits: dict[str, torch.Tensor],
        oof_targets: dict[str, torch.Tensor],
    ) -> None:
        if not self.calibrate:
            return

        prediction_heads = getattr(model, "prediction_heads", None)
        if prediction_heads is None:
            return

        for task, prediction_head in prediction_heads.items():
            if prediction_head is None or not getattr(prediction_head, "is_active", True):
                continue

            calibrator = getattr(prediction_head, "calibrator", None)
            decision_module = getattr(prediction_head, "decision_module", None)
            needs_calibrator_fit = calibrator is not None and getattr(calibrator, "is_active", True)
            needs_decision_fit = decision_module is not None and getattr(decision_module, "is_trainable", False)

            if not needs_calibrator_fit and not needs_decision_fit:
                continue

            if task not in oof_logits or task not in oof_targets:
                raise ValueError(
                    f"Post-hoc module for task {task!r} requires OOF logits/targets, but they are missing."
                )

            logits = oof_logits[task]
            targets = oof_targets[task]

            if needs_calibrator_fit:
                calibrator.fit(
                    logits=logits,
                    targets=targets,
                )
                calibrator.enable()

            if needs_decision_fit:
                probability_mapper = getattr(prediction_head, "probability_mapper", None)
                if probability_mapper is None:
                    raise ValueError(
                        f"Decision module for task {task!r} cannot be fit without a probability_mapper."
                    )

                with torch.no_grad():
                    logits_for_decision = logits
                    if calibrator is not None and getattr(calibrator, "is_active", True):
                        logits_for_decision = logits_for_decision.to(_module_device(calibrator))
                        logits_for_decision = calibrator(logits_for_decision)
                    else:
                        logits_for_decision = logits_for_decision.cpu()
                    probs = probability_mapper(logits_for_decision)

                decision_module.fit(
                    probs=probs.detach().cpu(),
                    targets=targets,
                )

    def _fit_calibrators_from_oof(
        self,
        model: Any,
        *,
        oof_logits: dict[str, torch.Tensor],
        oof_targets: dict[str, torch.Tensor],
    ) -> None:
        self._fit_posthoc_modules_from_oof(
            model,
            oof_logits=oof_logits,
            oof_targets=oof_targets,
        )

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

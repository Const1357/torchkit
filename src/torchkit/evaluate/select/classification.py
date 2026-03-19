from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor

from torchkit.evaluate.select._selector_evaluator import SelectorEvaluator
from torchkit.evaluate.mixins._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class _ClassificationSelectorBase(CalibratedLogitsFallbackMixin, SelectorEvaluator):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str,
        direction: str,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, direction=direction, weight=weight)
        self.score_key = score_key
        self.target_key = target_key
        self.probabilities_key = probabilities_key
        self.predictions_key = predictions_key

    @property
    def required_keys(self) -> tuple[str, ...]:
        keys = [self.score_key, self.target_key]
        if self.probabilities_key is not None:
            keys.append(self.probabilities_key)
        if self.predictions_key is not None:
            keys.append(self.predictions_key)
        return tuple(keys)

    @staticmethod
    def _scores_to_probabilities(scores: Tensor) -> Tensor:
        if scores.ndim == 1:
            p1 = torch.sigmoid(scores)
            return torch.stack([1.0 - p1, p1], dim=1)
        if scores.ndim == 2 and scores.shape[1] == 1:
            p1 = torch.sigmoid(scores[:, 0])
            return torch.stack([1.0 - p1, p1], dim=1)
        if scores.ndim == 2:
            return torch.softmax(scores, dim=1)
        raise ValueError(
            f"score tensor must be shape (N,), (N,1), or (N,C), got {tuple(scores.shape)}"
        )

    @staticmethod
    def _normalize_probabilities(probs: Tensor) -> Tensor:
        if probs.ndim == 1:
            return torch.stack([1.0 - probs, probs], dim=1)
        if probs.ndim == 2 and probs.shape[1] == 1:
            p1 = probs[:, 0]
            return torch.stack([1.0 - p1, p1], dim=1)
        if probs.ndim == 2:
            return probs
        raise ValueError(
            f"probabilities must be shape (N,), (N,1), or (N,C), got {tuple(probs.shape)}"
        )

    def _classification_tensors(
        self,
        inputs: dict[str, Any],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        scores = self._resolve_score_tensor(inputs)
        targets = self.resolve(inputs, self.target_key).detach()

        if targets.ndim != 1:
            raise ValueError(f"targets must be shape (N,), got {tuple(targets.shape)}")

        if self.probabilities_key is not None:
            probs_raw = self.resolve(inputs, self.probabilities_key).detach()
            probs = self._normalize_probabilities(probs_raw)
        else:
            probs = self._scores_to_probabilities(scores)

        if probs.ndim != 2:
            raise ValueError(f"probabilities must resolve to shape (N,C), got {tuple(probs.shape)}")

        n, c = probs.shape
        if targets.shape[0] != n:
            raise ValueError(
                f"targets batch size {targets.shape[0]} does not match probability batch size {n}"
            )

        if self.predictions_key is not None:
            preds = self.resolve(inputs, self.predictions_key).detach()
            if preds.ndim != 1:
                raise ValueError(f"predictions must be shape (N,), got {tuple(preds.shape)}")
            if preds.shape[0] != n:
                raise ValueError(
                    f"predictions batch size {preds.shape[0]} does not match probability batch size {n}"
                )
            preds = preds.long()
        else:
            preds = torch.argmax(probs, dim=1)

        device = probs.device
        dtype = probs.dtype
        cm = torch.zeros(c, c, device=device, dtype=torch.long)
        for t, p in zip(targets, preds):
            cm[t, p] += 1

        tp = cm.diag().to(dtype)
        fp = cm.sum(0).to(dtype) - tp
        fn = cm.sum(1).to(dtype) - tp
        support = cm.sum(1).to(dtype)
        eps = torch.tensor(1e-12, device=device, dtype=dtype)

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return probs, targets, preds, cm, precision, recall, f1, support


class AccuracySelectorEvaluator(_ClassificationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "accuracy",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, targets, preds, _, _, _, _, _ = self._classification_tensors(inputs)
        return (preds == targets).float().mean()


class BalancedAccuracySelectorEvaluator(_ClassificationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "balanced_accuracy",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, _, _, _, _, recall, _, _ = self._classification_tensors(inputs)
        return recall.mean()


class MacroPrecisionSelectorEvaluator(_ClassificationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "macro_precision",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, _, _, _, precision, _, _, _ = self._classification_tensors(inputs)
        return precision.mean()


class MacroRecallSelectorEvaluator(_ClassificationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "macro_recall",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, _, _, _, _, recall, _, _ = self._classification_tensors(inputs)
        return recall.mean()


class MacroF1SelectorEvaluator(_ClassificationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "macro_f1",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, _, _, _, _, _, f1, _ = self._classification_tensors(inputs)
        return f1.mean()


class MicroPrecisionSelectorEvaluator(_ClassificationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "micro_precision",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, _, _, cm, _, _, _, _ = self._classification_tensors(inputs)
        dtype = torch.float32 if not cm.is_floating_point() else cm.dtype
        tp = cm.diag().to(dtype).sum()
        fp = (cm.sum(0).to(dtype) - cm.diag().to(dtype)).sum()
        eps = torch.tensor(1e-12, device=cm.device, dtype=dtype)
        return tp / (tp + fp + eps)


class MicroRecallSelectorEvaluator(_ClassificationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "micro_recall",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, _, _, cm, _, _, _, _ = self._classification_tensors(inputs)
        dtype = torch.float32 if not cm.is_floating_point() else cm.dtype
        tp = cm.diag().to(dtype).sum()
        fn = (cm.sum(1).to(dtype) - cm.diag().to(dtype)).sum()
        eps = torch.tensor(1e-12, device=cm.device, dtype=dtype)
        return tp / (tp + fn + eps)


class MicroF1SelectorEvaluator(_ClassificationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "micro_f1",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        _, _, _, cm, _, _, _, _ = self._classification_tensors(inputs)
        dtype = torch.float32 if not cm.is_floating_point() else cm.dtype
        tp = cm.diag().to(dtype).sum()
        fp = (cm.sum(0).to(dtype) - cm.diag().to(dtype)).sum()
        fn = (cm.sum(1).to(dtype) - cm.diag().to(dtype)).sum()
        eps = torch.tensor(1e-12, device=cm.device, dtype=dtype)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        return 2 * precision * recall / (precision + recall + eps)


class BinaryPRAUCSelectorEvaluator(_ClassificationSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "pr_auc",
        weight: float = 1.0,
        n_points: int = 100,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            name=name,
            direction="maximize",
            weight=weight,
        )
        self.n_points = int(n_points)

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        probs, targets, _, _, _, _, _, _ = self._classification_tensors(inputs)
        if probs.shape[1] != 2:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)

        pos_probs = probs[:, 1]
        thresholds = torch.linspace(0, 1, self.n_points, device=probs.device, dtype=probs.dtype)
        eps = torch.tensor(1e-12, device=probs.device, dtype=probs.dtype)

        pr_curve_precision = torch.zeros(self.n_points, device=probs.device, dtype=probs.dtype)
        pr_curve_recall = torch.zeros(self.n_points, device=probs.device, dtype=probs.dtype)

        for i, thr in enumerate(thresholds):
            pred_pos = pos_probs >= thr
            tp_t = ((pred_pos) & (targets == 1)).sum().to(probs.dtype)
            fp_t = ((pred_pos) & (targets == 0)).sum().to(probs.dtype)
            fn_t = ((~pred_pos) & (targets == 1)).sum().to(probs.dtype)

            pr_curve_precision[i] = tp_t / (tp_t + fp_t + eps)
            pr_curve_recall[i] = tp_t / (tp_t + fn_t + eps)

        return torch.trapz(pr_curve_precision, pr_curve_recall)
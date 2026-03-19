from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from torchkit.evaluate.select._selector_evaluator import SelectorEvaluator
from torchkit.evaluate.mixins._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class _Segmentation2DSelectorBase(CalibratedLogitsFallbackMixin, SelectorEvaluator):
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
    def _derive_predictions_from_scores(scores: Tensor) -> tuple[Tensor, int]:
        if scores.ndim != 4:
            raise ValueError(f"scores must be (B,C,H,W), got {tuple(scores.shape)}")
        _, c, _, _ = scores.shape
        if c == 1:
            probs = torch.sigmoid(scores)
            preds = (probs > 0.5).long().squeeze(1)
            report_c = 2
        else:
            preds = torch.argmax(scores, dim=1)
            report_c = c
        return preds, report_c

    @staticmethod
    def _derive_predictions_from_probabilities(probs: Tensor) -> tuple[Tensor, int]:
        if probs.ndim != 4:
            raise ValueError(f"probabilities must be (B,C,H,W), got {tuple(probs.shape)}")
        _, c, _, _ = probs.shape
        if c == 1:
            preds = (probs > 0.5).long().squeeze(1)
            report_c = 2
        else:
            preds = torch.argmax(probs, dim=1)
            report_c = c
        return preds, report_c

    def _segmentation_tensors(self, inputs: dict[str, Any]) -> tuple[Tensor, Tensor, int]:
        scores = self._resolve_score_tensor(inputs)
        targets = self.resolve(inputs, self.target_key).detach()

        if scores.ndim != 4:
            raise ValueError("score tensor must be (B,C,H,W)")
        if targets.ndim != 3:
            raise ValueError("targets must be (B,H,W)")

        if self.predictions_key is not None:
            preds = self.resolve(inputs, self.predictions_key).detach()
            if preds.ndim != 3:
                raise ValueError(f"predictions must be (B,H,W), got {tuple(preds.shape)}")
            if preds.shape != targets.shape:
                raise ValueError(
                    f"predictions shape {tuple(preds.shape)} does not match targets shape {tuple(targets.shape)}"
                )
            score_c = scores.shape[1]
            report_c = 2 if score_c == 1 else score_c
        elif self.probabilities_key is not None:
            probs = self.resolve(inputs, self.probabilities_key).detach()
            preds, report_c = self._derive_predictions_from_probabilities(probs)
        else:
            preds, report_c = self._derive_predictions_from_scores(scores)

        return preds, targets, report_c

    @staticmethod
    def _perclass_stats(preds: Tensor, targets: Tensor, report_c: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        eps = 1e-12
        dices = []
        ious = []
        precisions = []
        recalls = []

        for cls_idx in range(report_c):
            pred_c = preds == cls_idx
            target_c = targets == cls_idx

            tp = (pred_c & target_c).sum().float()
            fp = (pred_c & ~target_c).sum().float()
            fn = (~pred_c & target_c).sum().float()

            dice = (2 * tp) / (2 * tp + fp + fn + eps)
            iou = tp / (tp + fp + fn + eps)
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)

            dices.append(dice)
            ious.append(iou)
            precisions.append(precision)
            recalls.append(recall)

        return (
            torch.stack(dices),
            torch.stack(ious),
            torch.stack(precisions),
            torch.stack(recalls),
        )


class SegmentationDiceSelectorEvaluator(_Segmentation2DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "dice",
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
        preds, targets, report_c = self._segmentation_tensors(inputs)
        dice, _, _, _ = self._perclass_stats(preds, targets, report_c)
        return dice.mean()


class SegmentationIoUSelectorEvaluator(_Segmentation2DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "iou",
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
        preds, targets, report_c = self._segmentation_tensors(inputs)
        _, iou, _, _ = self._perclass_stats(preds, targets, report_c)
        return iou.mean()


class SegmentationPrecisionSelectorEvaluator(_Segmentation2DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "precision",
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
        preds, targets, report_c = self._segmentation_tensors(inputs)
        _, _, precision, _ = self._perclass_stats(preds, targets, report_c)
        return precision.mean()


class SegmentationRecallSelectorEvaluator(_Segmentation2DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "recall",
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
        preds, targets, report_c = self._segmentation_tensors(inputs)
        _, _, _, recall = self._perclass_stats(preds, targets, report_c)
        return recall.mean()


class SegmentationPixelAccuracySelectorEvaluator(_Segmentation2DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "pixel_accuracy",
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
        preds, targets, _ = self._segmentation_tensors(inputs)
        return (preds == targets).float().mean()


class _Segmentation3DSelectorBase(CalibratedLogitsFallbackMixin, SelectorEvaluator):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str,
        direction: str,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name=name, direction=direction, weight=weight)
        self.score_key = score_key
        self.target_key = target_key
        self.probabilities_key = probabilities_key
        self.predictions_key = predictions_key
        self.spacing = voxel_spacing
        self.include_background = include_background

    @property
    def required_keys(self) -> tuple[str, ...]:
        keys = [self.score_key, self.target_key]
        if self.probabilities_key is not None:
            keys.append(self.probabilities_key)
        if self.predictions_key is not None:
            keys.append(self.predictions_key)
        return tuple(keys)

    @property
    def optional_keys(self) -> tuple[str, ...]:
        return (self.target_key,)

    @staticmethod
    def _derive_predictions_from_scores(scores: Tensor) -> tuple[Tensor, int]:
        if scores.ndim != 5:
            raise ValueError(f"scores must be (B,C,D,H,W), got {tuple(scores.shape)}")
        _, c, _, _, _ = scores.shape
        if c == 1:
            probs = torch.sigmoid(scores)
            preds = (probs > 0.5).long().squeeze(1)
            report_c = 2
        else:
            preds = torch.argmax(scores, dim=1)
            report_c = c
        return preds, report_c

    @staticmethod
    def _derive_predictions_from_probabilities(probs: Tensor) -> tuple[Tensor, int]:
        if probs.ndim != 5:
            raise ValueError(f"probabilities must be (B,C,D,H,W), got {tuple(probs.shape)}")
        _, c, _, _, _ = probs.shape
        if c == 1:
            preds = (probs > 0.5).long().squeeze(1)
            report_c = 2
        else:
            preds = torch.argmax(probs, dim=1)
            report_c = c
        return preds, report_c

    def _preds_targets_classes(self, inputs: dict[str, Any]) -> tuple[Tensor, Tensor | None, range]:
        scores_t = self._resolve_score_tensor(inputs)
        targets_t = self.resolve(inputs, self.target_key, strict=False)
        if targets_t is not None:
            targets_t = targets_t.detach()

        if scores_t.ndim != 5:
            raise ValueError("score tensor must be (B,C,D,H,W)")

        score_c = scores_t.shape[1]
        report_c = 2 if score_c == 1 else score_c

        classes = range(report_c)
        if not self.include_background:
            classes = range(1, report_c)

        if targets_t is None:
            return torch.empty(0), None, classes

        if targets_t.ndim != 4:
            raise ValueError("targets must be (B,D,H,W)")

        if self.predictions_key is not None:
            preds_t = self.resolve(inputs, self.predictions_key).detach()
            if preds_t.ndim != 4:
                raise ValueError(f"predictions must be (B,D,H,W), got {tuple(preds_t.shape)}")
            if preds_t.shape != targets_t.shape:
                raise ValueError(
                    f"predictions shape {tuple(preds_t.shape)} does not match targets shape {tuple(targets_t.shape)}"
                )
            preds = preds_t.cpu()
        elif self.probabilities_key is not None:
            probs_t = self.resolve(inputs, self.probabilities_key).detach()
            preds, _ = self._derive_predictions_from_probabilities(probs_t)
            preds = preds.cpu()
        else:
            preds, _ = self._derive_predictions_from_scores(scores_t)
            preds = preds.cpu()

        return preds, targets_t.cpu(), classes

    def _surface_distances(self, pred, target):
        import numpy as np
        from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure

        if pred.sum() == 0 and target.sum() == 0:
            return None

        struct = generate_binary_structure(rank=3, connectivity=1)
        pred_er = binary_erosion(pred, structure=struct, border_value=0)
        tgt_er = binary_erosion(target, structure=struct, border_value=0)

        pred_surface = np.logical_xor(pred, pred_er)
        tgt_surface = np.logical_xor(target, tgt_er)

        if pred_surface.sum() == 0 and tgt_surface.sum() == 0:
            return None

        dt_pred = distance_transform_edt(~pred_surface, sampling=self.spacing)
        dt_tgt = distance_transform_edt(~tgt_surface, sampling=self.spacing)

        dist_pred = dt_tgt[pred_surface] if pred_surface.sum() > 0 else np.array([], dtype=np.float64)
        dist_tgt = dt_pred[tgt_surface] if tgt_surface.sum() > 0 else np.array([], dtype=np.float64)

        if dist_pred.size == 0 and dist_tgt.size == 0:
            return None

        return np.concatenate([dist_pred, dist_tgt], axis=0)


class Segmentation3DDiceSelectorEvaluator(_Segmentation3DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str = "dice3d",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets, classes = self._preds_targets_classes(inputs)
        if targets is None:
            return torch.tensor(float("nan"))

        preds_np = preds.numpy()
        targets_np = targets.numpy()
        eps = 1e-12
        vals = []

        for b in range(preds_np.shape[0]):
            pred_b = preds_np[b]
            tgt_b = targets_np[b]
            for cls_idx in classes:
                pred = pred_b == cls_idx
                target = tgt_b == cls_idx
                tp = (pred & target).sum()
                fp = (pred & ~target).sum()
                fn = (~pred & target).sum()
                vals.append((2.0 * tp) / (2.0 * tp + fp + fn + eps))

        return torch.tensor(float(sum(vals) / len(vals))) if vals else torch.tensor(float("nan"))


class Segmentation3DIoUSelectorEvaluator(_Segmentation3DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str = "iou3d",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets, classes = self._preds_targets_classes(inputs)
        if targets is None:
            return torch.tensor(float("nan"))

        preds_np = preds.numpy()
        targets_np = targets.numpy()
        eps = 1e-12
        vals = []

        for b in range(preds_np.shape[0]):
            pred_b = preds_np[b]
            tgt_b = targets_np[b]
            for cls_idx in classes:
                pred = pred_b == cls_idx
                target = tgt_b == cls_idx
                tp = (pred & target).sum()
                fp = (pred & ~target).sum()
                fn = (~pred & target).sum()
                vals.append(tp / (tp + fp + fn + eps))

        return torch.tensor(float(sum(vals) / len(vals))) if vals else torch.tensor(float("nan"))


class Segmentation3DPrecisionSelectorEvaluator(_Segmentation3DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str = "precision3d",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets, classes = self._preds_targets_classes(inputs)
        if targets is None:
            return torch.tensor(float("nan"))

        preds_np = preds.numpy()
        targets_np = targets.numpy()
        eps = 1e-12
        vals = []

        for b in range(preds_np.shape[0]):
            pred_b = preds_np[b]
            tgt_b = targets_np[b]
            for cls_idx in classes:
                pred = pred_b == cls_idx
                target = tgt_b == cls_idx
                tp = (pred & target).sum()
                fp = (pred & ~target).sum()
                vals.append(tp / (tp + fp + eps))

        return torch.tensor(float(sum(vals) / len(vals))) if vals else torch.tensor(float("nan"))


class Segmentation3DRecallSelectorEvaluator(_Segmentation3DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str = "recall3d",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets, classes = self._preds_targets_classes(inputs)
        if targets is None:
            return torch.tensor(float("nan"))

        preds_np = preds.numpy()
        targets_np = targets.numpy()
        eps = 1e-12
        vals = []

        for b in range(preds_np.shape[0]):
            pred_b = preds_np[b]
            tgt_b = targets_np[b]
            for cls_idx in classes:
                pred = pred_b == cls_idx
                target = tgt_b == cls_idx
                tp = (pred & target).sum()
                fn = (~pred & target).sum()
                vals.append(tp / (tp + fn + eps))

        return torch.tensor(float(sum(vals) / len(vals))) if vals else torch.tensor(float("nan"))


class Segmentation3DVolumeSimilaritySelectorEvaluator(_Segmentation3DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str = "volume_similarity3d",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets, classes = self._preds_targets_classes(inputs)
        if targets is None:
            return torch.tensor(float("nan"))

        preds_np = preds.numpy()
        targets_np = targets.numpy()
        eps = 1e-12
        vals = []

        for b in range(preds_np.shape[0]):
            pred_b = preds_np[b]
            tgt_b = targets_np[b]
            for cls_idx in classes:
                pred = pred_b == cls_idx
                target = tgt_b == cls_idx
                pred_vol = pred.sum()
                target_vol = target.sum()
                vals.append(1.0 - (abs(pred_vol - target_vol) / (pred_vol + target_vol + eps)))

        return torch.tensor(float(sum(vals) / len(vals))) if vals else torch.tensor(float("nan"))


class Segmentation3DVoxelAccuracySelectorEvaluator(_Segmentation3DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str = "voxel_accuracy3d",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            name=name,
            direction="maximize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets, _ = self._preds_targets_classes(inputs)
        if targets is None:
            return torch.tensor(float("nan"))
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        return torch.tensor(float((preds_np == targets_np).mean()))


class Segmentation3DHD95SelectorEvaluator(_Segmentation3DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str = "hd95",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            name=name,
            direction="minimize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        import numpy as np

        preds, targets, classes = self._preds_targets_classes(inputs)
        if targets is None:
            return torch.tensor(float("nan"))

        preds_np = preds.numpy()
        targets_np = targets.numpy()
        vals = []

        for b in range(preds_np.shape[0]):
            pred_b = preds_np[b]
            tgt_b = targets_np[b]
            for cls_idx in classes:
                pred = pred_b == cls_idx
                target = tgt_b == cls_idx
                dist = self._surface_distances(pred, target)
                if dist is not None and dist.size > 0:
                    vals.append(float(np.percentile(dist, 95)))

        return torch.tensor(float(sum(vals) / len(vals))) if vals else torch.tensor(float("nan"))


class Segmentation3DASDSelectorEvaluator(_Segmentation3DSelectorBase):
    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str = "asd",
        weight: float = 1.0,
    ) -> None:
        super().__init__(
            score_key=score_key,
            target_key=target_key,
            probabilities_key=probabilities_key,
            predictions_key=predictions_key,
            voxel_spacing=voxel_spacing,
            include_background=include_background,
            name=name,
            direction="minimize",
            weight=weight,
        )

    def primary_metric(self, *, inputs: dict[str, Any]) -> Tensor:
        preds, targets, classes = self._preds_targets_classes(inputs)
        if targets is None:
            return torch.tensor(float("nan"))

        preds_np = preds.numpy()
        targets_np = targets.numpy()
        vals = []

        for b in range(preds_np.shape[0]):
            pred_b = preds_np[b]
            tgt_b = targets_np[b]
            for cls_idx in classes:
                pred = pred_b == cls_idx
                target = tgt_b == cls_idx
                dist = self._surface_distances(pred, target)
                if dist is not None and dist.size > 0:
                    vals.append(float(dist.mean()))

        return torch.tensor(float(sum(vals) / len(vals))) if vals else torch.tensor(float("nan"))
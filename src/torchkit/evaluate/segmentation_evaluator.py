from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from torchkit.evaluate._evaluator import Evaluator, MetricDirection
from torchkit.evaluate._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class SegmentationEvaluator(CalibratedLogitsFallbackMixin, Evaluator):
    """
    Segmentation evaluation.

    Supports binary and multiclass segmentation.

    Expected inputs:
        - score tensor:        (B,C,H,W) or (B,1,H,W)
        - targets:             (B,H,W)
        - optionally probabilities: (B,C,H,W) or (B,1,H,W)
        - optionally predictions:   (B,H,W)
    """

    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        name: str = "segmentation",
        primary_metric: str = "dice",
        direction: MetricDirection = "maximize",
        weight: float = 1.0,
    ):
        super().__init__(
            name=name,
            primary_metric=primary_metric,
            direction=direction,
            weight=weight,
        )

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
    def _derive_predictions_from_scores(scores: torch.Tensor) -> tuple[torch.Tensor, int]:
        if scores.ndim != 4:
            raise ValueError(f"scores must be (B,C,H,W), got {tuple(scores.shape)}")

        B, C, H, W = scores.shape
        if C == 1:
            probs = torch.sigmoid(scores)
            preds = (probs > 0.5).long().squeeze(1)
            report_C = 2
        else:
            preds = torch.argmax(scores, dim=1)
            report_C = C

        return preds, report_C

    @staticmethod
    def _derive_predictions_from_probabilities(probs: torch.Tensor) -> tuple[torch.Tensor, int]:
        if probs.ndim != 4:
            raise ValueError(f"probabilities must be (B,C,H,W), got {tuple(probs.shape)}")

        B, C, H, W = probs.shape
        if C == 1:
            preds = (probs > 0.5).long().squeeze(1)
            report_C = 2
        else:
            preds = torch.argmax(probs, dim=1)
            report_C = C

        return preds, report_C

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:

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

            score_C = scores.shape[1]
            report_C = 2 if score_C == 1 else score_C

        elif self.probabilities_key is not None:
            probs = self.resolve(inputs, self.probabilities_key).detach()
            preds, report_C = self._derive_predictions_from_probabilities(probs)

        else:
            preds, report_C = self._derive_predictions_from_scores(scores)

        metrics: dict[str, Any] = {}

        dice_list = []
        iou_list = []

        eps = 1e-12

        for c in range(report_C):
            pred_c = preds == c
            target_c = targets == c

            tp = (pred_c & target_c).sum().float()
            fp = (pred_c & ~target_c).sum().float()
            fn = (~pred_c & target_c).sum().float()

            dice = (2 * tp) / (2 * tp + fp + fn + eps)
            iou = tp / (tp + fp + fn + eps)

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)

            metrics[f"dice/class_{c}"] = float(dice)
            metrics[f"iou/class_{c}"] = float(iou)
            metrics[f"precision/class_{c}"] = float(precision)
            metrics[f"recall/class_{c}"] = float(recall)

            dice_list.append(dice)
            iou_list.append(iou)

        dice_tensor = torch.stack(dice_list)
        iou_tensor = torch.stack(iou_list)

        pixel_accuracy = (preds == targets).float().mean()

        metrics.update({
            "dice": float(dice_tensor.mean()),
            "iou": float(iou_tensor.mean()),
            "pixel_accuracy": float(pixel_accuracy),
        })

        return metrics


class Segmentation3DEvaluator(CalibratedLogitsFallbackMixin, Evaluator):
    """
    3D segmentation evaluator that handles optional masks.

    Metrics (stable keys; values may be None if targets are missing):
        dice, iou, precision, recall, volume_similarity, voxel_accuracy, hd95, asd
        plus per-class versions under ".../class_{c}"
    """

    def __init__(
        self,
        *,
        score_key: str,
        target_key: str,
        probabilities_key: Optional[str] = None,
        predictions_key: Optional[str] = None,
        voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        include_background: bool = False,
        name: str = "segmentation3d",
        primary_metric: str = "dice",
        direction: MetricDirection = "maximize",
        weight: float = 1.0,
    ):
        super().__init__(
            name=name,
            primary_metric=primary_metric,
            direction=direction,
            weight=weight,
        )

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
    def _derive_predictions_from_scores(scores: torch.Tensor) -> tuple[torch.Tensor, int]:
        if scores.ndim != 5:
            raise ValueError(f"scores must be (B,C,D,H,W), got {tuple(scores.shape)}")

        B, C, D, H, W = scores.shape
        if C == 1:
            probs = torch.sigmoid(scores)
            preds = (probs > 0.5).long().squeeze(1)
            report_C = 2
        else:
            preds = torch.argmax(scores, dim=1)
            report_C = C

        return preds, report_C

    @staticmethod
    def _derive_predictions_from_probabilities(probs: torch.Tensor) -> tuple[torch.Tensor, int]:
        if probs.ndim != 5:
            raise ValueError(f"probabilities must be (B,C,D,H,W), got {tuple(probs.shape)}")

        B, C, D, H, W = probs.shape
        if C == 1:
            preds = (probs > 0.5).long().squeeze(1)
            report_C = 2
        else:
            preds = torch.argmax(probs, dim=1)
            report_C = C

        return preds, report_C

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

    def metrics(self, *, inputs: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        scores_t = self._resolve_score_tensor(inputs)

        targets_t = self.resolve(inputs, self.target_key, strict=False)
        if targets_t is not None:
            targets_t = targets_t.detach()

        if scores_t.ndim != 5:
            raise ValueError("score tensor must be (B,C,D,H,W)")

        score_C = scores_t.shape[1]
        report_C = 2 if score_C == 1 else score_C

        classes = range(report_C)
        if not self.include_background:
            classes = range(1, report_C)

        metrics: dict[str, Any] = {}

        for c in classes:
            metrics[f"dice/class_{c}"] = None
            metrics[f"iou/class_{c}"] = None
            metrics[f"precision/class_{c}"] = None
            metrics[f"recall/class_{c}"] = None
            metrics[f"volume_similarity/class_{c}"] = None
            metrics[f"hd95/class_{c}"] = None
            metrics[f"asd/class_{c}"] = None

        metrics["dice"] = None
        metrics["iou"] = None
        metrics["precision"] = None
        metrics["recall"] = None
        metrics["volume_similarity"] = None
        metrics["hd95"] = None
        metrics["asd"] = None
        metrics["voxel_accuracy"] = None

        if targets_t is None:
            return metrics

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

        targets = targets_t.cpu()

        preds_np = preds.numpy()
        targets_np = targets.numpy()

        eps = 1e-12

        dice_vals = {c: [] for c in classes}
        iou_vals = {c: [] for c in classes}
        prec_vals = {c: [] for c in classes}
        rec_vals = {c: [] for c in classes}
        vs_vals = {c: [] for c in classes}
        hd95_vals = {c: [] for c in classes}
        asd_vals = {c: [] for c in classes}

        B = preds_np.shape[0]

        for b in range(B):
            pred_b = preds_np[b]
            tgt_b = targets_np[b]

            for c in classes:
                pred = (pred_b == c)
                target = (tgt_b == c)

                tp = np.logical_and(pred, target).sum()
                fp = np.logical_and(pred, np.logical_not(target)).sum()
                fn = np.logical_and(np.logical_not(pred), target).sum()

                dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
                iou = tp / (tp + fp + fn + eps)

                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)

                pred_vol = pred.sum()
                target_vol = target.sum()
                vs = 1.0 - (abs(pred_vol - target_vol) / (pred_vol + target_vol + eps))

                dice_vals[c].append(dice)
                iou_vals[c].append(iou)
                prec_vals[c].append(precision)
                rec_vals[c].append(recall)
                vs_vals[c].append(vs)

                dist = self._surface_distances(pred, target)
                if dist is not None and dist.size > 0:
                    hd95_vals[c].append(float(np.percentile(dist, 95)))
                    asd_vals[c].append(float(dist.mean()))

        perclass_dice = []
        perclass_iou = []
        perclass_prec = []
        perclass_rec = []
        perclass_vs = []
        perclass_hd95 = []
        perclass_asd = []

        for c in classes:
            d = float(np.mean(dice_vals[c])) if dice_vals[c] else None
            j = float(np.mean(iou_vals[c])) if iou_vals[c] else None
            p = float(np.mean(prec_vals[c])) if prec_vals[c] else None
            r = float(np.mean(rec_vals[c])) if rec_vals[c] else None
            v = float(np.mean(vs_vals[c])) if vs_vals[c] else None

            metrics[f"dice/class_{c}"] = d
            metrics[f"iou/class_{c}"] = j
            metrics[f"precision/class_{c}"] = p
            metrics[f"recall/class_{c}"] = r
            metrics[f"volume_similarity/class_{c}"] = v

            if d is not None:
                perclass_dice.append(d)
            if j is not None:
                perclass_iou.append(j)
            if p is not None:
                perclass_prec.append(p)
            if r is not None:
                perclass_rec.append(r)
            if v is not None:
                perclass_vs.append(v)

            if hd95_vals[c]:
                h = float(np.mean(hd95_vals[c]))
                a = float(np.mean(asd_vals[c]))
                metrics[f"hd95/class_{c}"] = h
                metrics[f"asd/class_{c}"] = a
                perclass_hd95.append(h)
                perclass_asd.append(a)
            else:
                metrics[f"hd95/class_{c}"] = None
                metrics[f"asd/class_{c}"] = None

        metrics["dice"] = float(np.mean(perclass_dice)) if perclass_dice else None
        metrics["iou"] = float(np.mean(perclass_iou)) if perclass_iou else None
        metrics["precision"] = float(np.mean(perclass_prec)) if perclass_prec else None
        metrics["recall"] = float(np.mean(perclass_rec)) if perclass_rec else None
        metrics["volume_similarity"] = float(np.mean(perclass_vs)) if perclass_vs else None
        metrics["hd95"] = float(np.mean(perclass_hd95)) if perclass_hd95 else None
        metrics["asd"] = float(np.mean(perclass_asd)) if perclass_asd else None

        metrics["voxel_accuracy"] = float((preds_np == targets_np).mean())

        return metrics
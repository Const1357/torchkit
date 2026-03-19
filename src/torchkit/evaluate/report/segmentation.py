from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from torchkit.evaluate.report._report_evaluator import ReportEvaluator
from torchkit.evaluate.mixins._calibrated_logits_fallback_mixin import CalibratedLogitsFallbackMixin


class SegmentationReportEvaluator(CalibratedLogitsFallbackMixin, ReportEvaluator):
    """
    Segmentation evaluation.

    Supports binary and multiclass segmentation.

    Expected inputs:
        - score tensor:             (B,C,H,W) or (B,1,H,W)
        - targets:                  (B,H,W)
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
    ) -> None:
        super().__init__(name=name)

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
    def _derive_predictions_from_probabilities(probs: torch.Tensor) -> tuple[torch.Tensor, int]:
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
            score_c = scores.shape[1]
            report_c = 2 if score_c == 1 else score_c
        elif self.probabilities_key is not None:
            probs = self.resolve(inputs, self.probabilities_key).detach()
            preds, report_c = self._derive_predictions_from_probabilities(probs)
        else:
            preds, report_c = self._derive_predictions_from_scores(scores)

        metrics: dict[str, Any] = {}
        dice_list = []
        iou_list = []
        eps = 1e-12

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

            metrics[f"dice/class_{cls_idx}"] = float(dice)
            metrics[f"iou/class_{cls_idx}"] = float(iou)
            metrics[f"precision/class_{cls_idx}"] = float(precision)
            metrics[f"recall/class_{cls_idx}"] = float(recall)

            dice_list.append(dice)
            iou_list.append(iou)

        dice_tensor = torch.stack(dice_list)
        iou_tensor = torch.stack(iou_list)
        pixel_accuracy = (preds == targets).float().mean()

        metrics.update(
            {
                "dice": float(dice_tensor.mean()),
                "iou": float(iou_tensor.mean()),
                "pixel_accuracy": float(pixel_accuracy),
            }
        )
        return metrics


class Segmentation3DReportEvaluator(CalibratedLogitsFallbackMixin, ReportEvaluator):
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
    ) -> None:
        super().__init__(name=name)

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
    def _derive_predictions_from_probabilities(probs: torch.Tensor) -> tuple[torch.Tensor, int]:
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

        score_c = scores_t.shape[1]
        report_c = 2 if score_c == 1 else score_c

        classes = range(report_c)
        if not self.include_background:
            classes = range(1, report_c)

        metrics: dict[str, Any] = {}

        for cls_idx in classes:
            metrics[f"dice/class_{cls_idx}"] = None
            metrics[f"iou/class_{cls_idx}"] = None
            metrics[f"precision/class_{cls_idx}"] = None
            metrics[f"recall/class_{cls_idx}"] = None
            metrics[f"volume_similarity/class_{cls_idx}"] = None
            metrics[f"hd95/class_{cls_idx}"] = None
            metrics[f"asd/class_{cls_idx}"] = None

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

        bsz = preds_np.shape[0]

        for b in range(bsz):
            pred_b = preds_np[b]
            tgt_b = targets_np[b]

            for cls_idx in classes:
                pred = pred_b == cls_idx
                target = tgt_b == cls_idx

                tp = (pred & target).sum()
                fp = (pred & ~target).sum()
                fn = (~pred & target).sum()

                dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
                iou = tp / (tp + fp + fn + eps)
                precision = tp / (tp + fp + eps)
                recall = tp / (tp + fn + eps)

                pred_vol = pred.sum()
                target_vol = target.sum()
                vs = 1.0 - (abs(pred_vol - target_vol) / (pred_vol + target_vol + eps))

                dice_vals[cls_idx].append(dice)
                iou_vals[cls_idx].append(iou)
                prec_vals[cls_idx].append(precision)
                rec_vals[cls_idx].append(recall)
                vs_vals[cls_idx].append(vs)

                dist = self._surface_distances(pred, target)
                if dist is not None and dist.size > 0:
                    hd95_vals[cls_idx].append(float(np.percentile(dist, 95)))
                    asd_vals[cls_idx].append(float(dist.mean()))

        perclass_dice = []
        perclass_iou = []
        perclass_prec = []
        perclass_rec = []
        perclass_vs = []
        perclass_hd95 = []
        perclass_asd = []

        for cls_idx in classes:
            d = float(np.mean(dice_vals[cls_idx])) if dice_vals[cls_idx] else None
            j = float(np.mean(iou_vals[cls_idx])) if iou_vals[cls_idx] else None
            p = float(np.mean(prec_vals[cls_idx])) if prec_vals[cls_idx] else None
            r = float(np.mean(rec_vals[cls_idx])) if rec_vals[cls_idx] else None
            v = float(np.mean(vs_vals[cls_idx])) if vs_vals[cls_idx] else None

            metrics[f"dice/class_{cls_idx}"] = d
            metrics[f"iou/class_{cls_idx}"] = j
            metrics[f"precision/class_{cls_idx}"] = p
            metrics[f"recall/class_{cls_idx}"] = r
            metrics[f"volume_similarity/class_{cls_idx}"] = v

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

            if hd95_vals[cls_idx]:
                h = float(np.mean(hd95_vals[cls_idx]))
                a = float(np.mean(asd_vals[cls_idx]))
                metrics[f"hd95/class_{cls_idx}"] = h
                metrics[f"asd/class_{cls_idx}"] = a
                perclass_hd95.append(h)
                perclass_asd.append(a)
            else:
                metrics[f"hd95/class_{cls_idx}"] = None
                metrics[f"asd/class_{cls_idx}"] = None

        metrics["dice"] = float(np.mean(perclass_dice)) if perclass_dice else None
        metrics["iou"] = float(np.mean(perclass_iou)) if perclass_iou else None
        metrics["precision"] = float(np.mean(perclass_prec)) if perclass_prec else None
        metrics["recall"] = float(np.mean(perclass_rec)) if perclass_rec else None
        metrics["volume_similarity"] = float(np.mean(perclass_vs)) if perclass_vs else None
        metrics["hd95"] = float(np.mean(perclass_hd95)) if perclass_hd95 else None
        metrics["asd"] = float(np.mean(perclass_asd)) if perclass_asd else None
        metrics["voxel_accuracy"] = float((preds_np == targets_np).mean())

        return metrics

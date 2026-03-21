from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from torchkit.models.decision._decision_module import DecisionModule
from torchkit.models._spec_utils import normalize_spec_kwargs


BinaryThresholdTuningMethod = Literal["youden_optimal", "scan", "none"]
BinaryThresholdScanMetric = Literal["accuracy", "precision", "recall", "f1", "balanced_accuracy"]


class BinaryClassificationThreshold(DecisionModule):
    """
    Binary classification decision module.

    Supported input shapes:
    - (N,)   : binary probabilities
    - (N, 1) : binary probabilities
    - (N, 2) : two-class probabilities

    Returns:
    - (N,) integer predictions in {0, 1}

    ### *Note*
    For (N, 2) input shape, the second column (index 1) is treated
        as the positive class probability.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        tuning_method: BinaryThresholdTuningMethod = "youden_optimal",
        tuning_metric: BinaryThresholdScanMetric = "balanced_accuracy",
        coarse_scan_points: int = 101,
        refined_scan_points: int = 201,
    ):
        super().__init__()
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
        if tuning_method not in {"youden_optimal", "scan", "none"}:
            raise ValueError(
                "tuning_method must be one of "
                "{'youden_optimal', 'scan', 'none'}, "
                f"got {tuning_method!r}."
            )
        if tuning_metric not in {"accuracy", "precision", "recall", "f1", "balanced_accuracy"}:
            raise ValueError(
                "tuning_metric must be one of "
                "{'accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy'}, "
                f"got {tuning_metric!r}."
            )
        if coarse_scan_points < 2:
            raise ValueError(f"coarse_scan_points must be >= 2, got {coarse_scan_points}.")
        if refined_scan_points < 2:
            raise ValueError(f"refined_scan_points must be >= 2, got {refined_scan_points}.")

        self._spec_kwargs = normalize_spec_kwargs(
            {
                "threshold": threshold,
                "tuning_method": tuning_method,
                "tuning_metric": tuning_metric,
                "coarse_scan_points": coarse_scan_points,
                "refined_scan_points": refined_scan_points,
            }
        )
        self.register_buffer("_threshold", torch.tensor(float(threshold), dtype=torch.float32))
        self.tuning_method = tuning_method
        self.tuning_metric = tuning_metric
        self.coarse_scan_points = int(coarse_scan_points)
        self.refined_scan_points = int(refined_scan_points)

    @property
    def threshold(self) -> float:
        return float(self._threshold.item())
    
    @threshold.setter
    def threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}.")
        self._threshold.fill_(float(value))

    @staticmethod
    def _extract_positive_probs(probs: Tensor) -> Tensor:
        if probs.ndim == 1:
            return probs

        if probs.ndim == 2 and probs.shape[1] == 1:
            return probs[:, 0]

        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]

        raise ValueError(
            "BinaryClassificationThreshold expects binary probabilities of shape "
            f"(N,), (N,1), or (N,2). Got shape {tuple(probs.shape)}."
        )

    @staticmethod
    def _extract_binary_targets(targets: Tensor) -> Tensor:
        if targets.ndim == 1:
            y = targets
        elif targets.ndim == 2 and targets.shape[1] == 1:
            y = targets[:, 0]
        elif targets.ndim == 2 and targets.shape[1] == 2:
            y = torch.argmax(targets, dim=1)
        else:
            raise ValueError(
                "BinaryClassificationThreshold fit expects binary targets of shape "
                f"(N,), (N,1), or (N,2). Got shape {tuple(targets.shape)}."
            )

        if torch.is_floating_point(y):
            y = (y >= 0.5).to(dtype=torch.long)
        else:
            y = y.to(dtype=torch.long)

        return y

    def _metric_score(self, y_true: Tensor, y_pred: Tensor) -> float:
        y_true = y_true.to(dtype=torch.long)
        y_pred = y_pred.to(dtype=torch.long)

        tp = torch.sum((y_true == 1) & (y_pred == 1)).to(dtype=torch.float32)
        tn = torch.sum((y_true == 0) & (y_pred == 0)).to(dtype=torch.float32)
        fp = torch.sum((y_true == 0) & (y_pred == 1)).to(dtype=torch.float32)
        fn = torch.sum((y_true == 1) & (y_pred == 0)).to(dtype=torch.float32)

        total = tp + tn + fp + fn
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        specificity = tn / (tn + fp).clamp_min(1.0)

        if self.tuning_metric == "accuracy":
            return float(((tp + tn) / total.clamp_min(1.0)).item())
        if self.tuning_metric == "precision":
            return float(precision.item())
        if self.tuning_metric == "recall":
            return float(recall.item())
        if self.tuning_metric == "f1":
            return float((2.0 * precision * recall / (precision + recall).clamp_min(1e-12)).item())
        if self.tuning_metric == "balanced_accuracy":
            return float((0.5 * (recall + specificity)).item())
        raise RuntimeError(f"Unsupported tuning_metric {self.tuning_metric!r}.")

    @staticmethod
    def _pick_best_threshold(
        thresholds: Tensor,
        scores: Tensor,
        reference_threshold: float,
    ) -> float:
        max_score = torch.max(scores)
        candidate_mask = scores == max_score
        candidate_thresholds = thresholds[candidate_mask]
        if candidate_thresholds.numel() == 1:
            return float(candidate_thresholds.item())

        reference = torch.tensor(
            float(reference_threshold),
            dtype=candidate_thresholds.dtype,
            device=candidate_thresholds.device,
        )
        best_idx = torch.argmin(torch.abs(candidate_thresholds - reference))
        return float(candidate_thresholds[best_idx].item())

    def _fit_youden_optimal(self, p_pos: Tensor, y_true: Tensor) -> float:
        if torch.unique(y_true).numel() < 2:
            return self.threshold

        thresholds_t = torch.unique(p_pos.detach()).sort().values
        positives = torch.sum(y_true == 1).to(dtype=torch.float32)
        negatives = torch.sum(y_true == 0).to(dtype=torch.float32)

        j_scores = []
        for threshold in thresholds_t:
            preds = (p_pos >= threshold).to(dtype=torch.long)
            tp = torch.sum((y_true == 1) & (preds == 1)).to(dtype=torch.float32)
            fp = torch.sum((y_true == 0) & (preds == 1)).to(dtype=torch.float32)
            tpr = tp / positives.clamp_min(1.0)
            fpr = fp / negatives.clamp_min(1.0)
            j_scores.append((tpr - fpr).item())

        j_scores_t = torch.tensor(j_scores, dtype=p_pos.dtype, device=thresholds_t.device)
        return self._pick_best_threshold(thresholds_t, j_scores_t, reference_threshold=self.threshold)

    def _fit_scan(self, p_pos: Tensor, y_true: Tensor) -> float:
        device = p_pos.device
        coarse_thresholds = torch.linspace(0.0, 1.0, self.coarse_scan_points, device=device, dtype=p_pos.dtype)

        coarse_scores = []
        for threshold in coarse_thresholds:
            preds = (p_pos >= threshold).to(dtype=torch.long)
            coarse_scores.append(self._metric_score(y_true, preds))

        coarse_scores_t = torch.tensor(coarse_scores, device=device, dtype=p_pos.dtype)
        best_coarse = self._pick_best_threshold(
            coarse_thresholds,
            coarse_scores_t,
            reference_threshold=self.threshold,
        )

        coarse_step = 1.0 / float(self.coarse_scan_points - 1)
        refined_low = max(0.0, best_coarse - coarse_step)
        refined_high = min(1.0, best_coarse + coarse_step)
        refined_thresholds = torch.linspace(
            refined_low,
            refined_high,
            self.refined_scan_points,
            device=device,
            dtype=p_pos.dtype,
        )

        refined_scores = []
        for threshold in refined_thresholds:
            preds = (p_pos >= threshold).to(dtype=torch.long)
            refined_scores.append(self._metric_score(y_true, preds))

        refined_scores_t = torch.tensor(refined_scores, device=device, dtype=p_pos.dtype)
        return self._pick_best_threshold(
            refined_thresholds,
            refined_scores_t,
            reference_threshold=best_coarse,
        )

    def forward_impl(self, probs: Tensor) -> Tensor:
        p_pos = self._extract_positive_probs(probs)
        return (p_pos >= self.threshold).to(dtype=torch.long)

    def fit_impl(self, probs: Tensor, targets: Tensor) -> None:
        if self.tuning_method == "none":
            return

        p_pos = self._extract_positive_probs(probs).detach()
        y_true = self._extract_binary_targets(targets).detach()

        if self.tuning_method == "youden_optimal":
            self.threshold = self._fit_youden_optimal(p_pos, y_true)
            return

        if self.tuning_method == "scan":
            self.threshold = self._fit_scan(p_pos, y_true)
            return

        raise RuntimeError(f"Unsupported tuning_method {self.tuning_method!r}.")

    def to_spec(self):
        from torchkit.models.decision.factory import DecisionModuleSpec

        return DecisionModuleSpec(
            cls=self.__class__,
            kwargs={
                "threshold": self.threshold,
                "tuning_method": self.tuning_method,
                "tuning_metric": self.tuning_metric,
                "coarse_scan_points": self.coarse_scan_points,
                "refined_scan_points": self.refined_scan_points,
            },
        )
    

class ArgmaxDecision(DecisionModule):
    """
    Multiclass classification decision module.

    Supported input shapes:
    - (N, C) with C >= 2 : multiclass probabilities

    Returns:
    - (N,) integer predictions in {0, 1, ..., C-1}
    """

    def forward_impl(self, probs: Tensor) -> Tensor:

        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError(
                f"{self.__class__.__name__} expects multiclass probabilities of shape (N, C) with C >= 2. "
                f"Got shape {tuple(probs.shape)}."
            )

        return torch.argmax(probs, dim=1)

    def to_spec(self):
        return super().to_spec()
    
class SampleTopKTemperature(DecisionModule):
    """
    Multiclass classification decision module that samples from top-k classes with temperature scaling.

    Supported input shapes:
    - (N, C) with C >= 2 : multiclass probabilities

    Returns:
    - (N,) integer predictions in {0, 1, ..., C-1}
    """

    def __init__(self, k: int = 5, temperature: float = 1.0):
        super().__init__()
        self._spec_kwargs = normalize_spec_kwargs(
            {
                "k": k,
                "temperature": temperature,
            }
        )

        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        self.k = int(k)
        self.temperature = float(temperature)

    def forward_impl(self, probs: Tensor) -> Tensor:

        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError(
                f"{self.__class__.__name__} expects multiclass probabilities of shape (N, C) with C >= 2. "
                f"Got shape {tuple(probs.shape)}."
            )

        # Apply temperature scaling
        scaled_probs = torch.pow(probs, 1.0 / self.temperature)

        # Get top-k indices
        topk_probs, topk_indices = torch.topk(scaled_probs, k=min(self.k, probs.shape[1]), dim=1)

        # Normalize top-k probabilities
        topk_probs_normalized = topk_probs / torch.sum(topk_probs, dim=1, keepdim=True)

        # Sample from the top-k distribution
        sampled_indices = torch.multinomial(topk_probs_normalized, num_samples=1).squeeze(1)

        # Map back to original class indices
        return topk_indices.gather(1, sampled_indices.unsqueeze(1)).squeeze(1)

    def to_spec(self):
        from torchkit.models.decision.factory import DecisionModuleSpec

        return DecisionModuleSpec(
            cls=self.__class__,
            kwargs={
                "k": self.k,
                "temperature": self.temperature,
            },
        )

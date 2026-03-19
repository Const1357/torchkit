from __future__ import annotations
from typing import Literal, Optional
from torch import Tensor
import torch

"""Definitions of Relational Objectives

Relational objectives operate between an input and a target."""

from torchkit.objectives._base import Objective

class BCELoss(Objective):

    def __init__(
        self,
        input_path: str,
        target_path: str,
        *,
        class_weight: Optional[Tensor] = None,
        name: str = "binary_cross_entropy_loss",
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        is_optional: bool = False,
    ):
        super().__init__(
            name=name,
            weight=weight,
            reduction=reduction,
            is_optional=is_optional,
        )

        self._input_path = input_path
        self._target_path = f"{target_path}"
        self._class_weight = class_weight

        self._required_keys = (input_path, target_path)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return self._required_keys
    
    def loss(self, *, inputs: dict[str, Tensor]) -> Tensor:
        # resolve the input and target tensors from the inputs dict
        input_tensor = self.resolve(inputs, self._input_path)
        target_tensor = self.resolve(inputs, self._target_path)

        from torch.nn.functional import binary_cross_entropy_with_logits

        return binary_cross_entropy_with_logits(
            input=input_tensor,
            target=target_tensor,
            weight=self._class_weight,
            reduction=self.reduction,
        )

class CELoss(Objective):

    def __init__(
        self,
        input_path: str,
        target_path: str,
        *,
        class_weight: Optional[Tensor] = None,
        name: str = "cross_entropy_loss",
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        is_optional: bool = False,
    ):
        super().__init__(
            name=name,
            weight=weight,
            reduction=reduction,
            is_optional=is_optional,
        )

        self._input_path = input_path
        self._target_path = f"{target_path}"
        self._class_weight = class_weight

        self._required_keys = (input_path, target_path)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return self._required_keys
    
    def loss(self, *, inputs: dict[str, Tensor]) -> Tensor:
        # resolve the input and target tensors from the inputs dict
        input_tensor = self.resolve(inputs, self._input_path)
        target_tensor = self.resolve(inputs, self._target_path)

        from torch.nn.functional import cross_entropy

        return cross_entropy(
            input=input_tensor,
            target=target_tensor,
            weight=self._class_weight,
            reduction=self.reduction,
        )
    

class MSELoss(Objective):
    
    def __init__(
        self,
        input_path: str,
        target_path: str,
        *,
        name: str = "mean_squared_error_loss",
        weight: float = 1.0,
        is_optional: bool = False,
        reduction: Literal["mean", "sum"] = "mean",
    ):
        super().__init__(
            name=name,
            weight=weight,
            is_optional=is_optional,
            reduction=reduction,
        )

        self._input_path = input_path
        self._target_path = f"{target_path}"

        self._required_keys = (input_path, target_path)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return self._required_keys

    def loss(self, *, inputs: dict[str, Tensor]) -> Tensor:
        # resolve the input and target tensors from the inputs dict
        input_tensor = self.resolve(inputs, self._input_path)
        target_tensor = self.resolve(inputs, self._target_path)

        from torch.nn.functional import mse_loss

        return mse_loss(
            input=input_tensor,
            target=target_tensor,
            reduction=self.reduction,
        )

# TODO: add more objective implementation. Use these as template.

class DiceLoss(Objective):
    """
    Dice loss with NaN-masked supervision support.

    Assumptions:
    - `logits` is a Tensor with shape (B, 1, *spatial) or (B, *spatial) for binary segmentation,
      OR (B, C, *spatial) for multiclass segmentation.
    - `mask` is a Tensor with the SAME spatial shape as the target, batched, and uses NaNs to mark
      samples with missing supervision.
      Example: for missing mask in sample i, mask[i] is entirely NaN.

    Behavior:
    - If some masks exist in the batch: compute loss only on valid samples.
    - If no masks exist in the batch:
        - if is_optional=True: returns zero loss
        - else: raises ValueError
    """

    def __init__(
        self,
        logits_path: str,
        mask_path: str,
        *,
        name: str = "dice_loss",
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        is_optional: bool = False,
        smooth: float = 1e-6,
        include_background: bool = True,
        from_logits: bool = True,
    ):
        super().__init__(
            name=name,
            weight=weight,
            reduction=reduction,
            is_optional=is_optional,
        )

        self._logits_path = logits_path
        self._mask_path = f"{mask_path}"

        self.smooth = float(smooth)
        self.include_background = bool(include_background)
        self.from_logits = bool(from_logits)

        self._required_keys = (self._logits_path, self._mask_path)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return self._required_keys

    @staticmethod
    def _valid_mask_indices(mask: Tensor) -> Tensor:
        """
        Returns a boolean tensor (B,) where True means the sample has a valid mask (not all-NaN).
        """
        if mask.ndim < 2:
            raise ValueError(f"mask must be batched with ndim>=2, got shape={tuple(mask.shape)}")
        B = mask.shape[0]
        # A sample is invalid if ALL entries are NaN.
        return ~torch.isnan(mask).view(B, -1).all(dim=1)

    def loss(self, *, inputs: dict[str, Tensor]) -> Tensor:
        logits = self.resolve(inputs, self._logits_path)
        mask = self.resolve(inputs, self._mask_path)

        if logits.ndim < 2:
            raise ValueError(f"logits must be batched, got shape={tuple(logits.shape)}")
        if mask.ndim < 2:
            raise ValueError(f"mask must be batched, got shape={tuple(mask.shape)}")

        # Determine valid supervised samples.
        valid = self._valid_mask_indices(mask)
        if not bool(valid.any().item()):
            if self.is_optional:
                return self._zero_loss(inputs)
            raise ValueError(
                f"Objective '{self.name}': no valid masks in this batch (all masks are NaN)."
            )

        # Filter only supervised samples.
        logits = logits[valid]
        mask = mask[valid]

        # Replace NaNs inside valid samples (should be none, but defensive)
        mask = torch.nan_to_num(mask, nan=0.0)

        eps = self.smooth

        # ------------------------
        # Binary segmentation
        # ------------------------
        # logits: (B, 1, *spatial) or (B, *spatial)
        # mask  : (B, 1, *spatial) or (B, *spatial)
        if logits.ndim == mask.ndim:
            # could be (B, *spatial) binary OR (B, C, *spatial) multiclass one-hot target.
            # We'll disambiguate by checking if logits has a channel dim and mask is integer labels.
            pass

        # If mask is integer labels (multiclass typical): (B, *spatial)
        # If mask is float/binary: (B, 1, *spatial) or (B, *spatial)
        is_multiclass = logits.ndim >= 3 and (logits.shape[1] > 1) and (mask.ndim == logits.ndim - 1)

        if is_multiclass:
            # ------------------------
            # Multiclass segmentation
            # logits: (B, C, *spatial)
            # mask  : (B, *spatial) with class indices
            # ------------------------
            if mask.dtype.is_floating_point:
                # you *can* allow float labels, but class indices should be integer
                # (float labels are typically invalid here)
                raise TypeError(
                    f"Multiclass Dice expects integer class-index targets, got floating mask dtype={mask.dtype}."
                )

            B, C = logits.shape[:2]
            spatial = logits.shape[2:]
            if tuple(mask.shape[1:]) != tuple(spatial):
                raise ValueError(
                    f"Shape mismatch: logits spatial={spatial}, mask spatial={tuple(mask.shape[1:])}."
                )

            if self.from_logits:
                probs = torch.softmax(logits, dim=1)
            else:
                probs = logits  # assume already probabilities

            # one-hot targets: (B, C, *spatial)
            tgt = torch.nn.functional.one_hot(mask.long(), num_classes=C)  # (B, *spatial, C)
            tgt = tgt.permute(0, -1, *range(1, tgt.ndim - 1)).contiguous().to(dtype=probs.dtype)

            # Optionally drop background channel 0
            if not self.include_background and C > 1:
                probs = probs[:, 1:]
                tgt = tgt[:, 1:]
                C_eff = C - 1
            else:
                C_eff = C

            probs_f = probs.reshape(B, C_eff, -1)
            tgt_f = tgt.reshape(B, C_eff, -1)

            inter = (probs_f * tgt_f).sum(dim=-1)
            denom = probs_f.sum(dim=-1) + tgt_f.sum(dim=-1)

            dice = (2.0 * inter + eps) / (denom + eps)  # (B, C_eff)
            loss_per_class = 1.0 - dice                # (B, C_eff)

            # reduce across class first
            loss_per_sample = loss_per_class.mean(dim=1)  # (B,)

        else:
            # ------------------------
            # Binary segmentation
            # logits: (B, 1, *spatial) or (B, *spatial)
            # mask  : (B, 1, *spatial) or (B, *spatial) in {0,1}
            # ------------------------
            if logits.ndim == mask.ndim - 1:
                raise ValueError(
                    f"Binary Dice expects logits and mask to have matching dims "
                    f"(or both with channel=1), got logits={tuple(logits.shape)}, mask={tuple(mask.shape)}."
                )

            if logits.ndim == 2:
                # (B, S) is also acceptable
                pass

            # ensure channel dim is present and =1 for binary
            if logits.ndim >= 3 and logits.shape[1] != 1:
                raise ValueError(
                    f"Binary Dice expects logits channel dim=1, got logits.shape[1]={logits.shape[1]}."
                )

            # align shapes
            if mask.ndim == logits.ndim - 1 and logits.ndim >= 3:
                # logits (B,1,...) ; mask (B,...)
                mask = mask.unsqueeze(1)
            elif mask.ndim == logits.ndim and logits.ndim >= 3:
                # both (B,1,...)
                pass
            elif logits.ndim == 2 and mask.ndim == 1:
                # logits (B,S), mask (B,) is invalid for segmentation
                raise ValueError("For segmentation, mask cannot be (B,) when logits is (B,S).")
            elif logits.ndim == 2 and mask.ndim == 2:
                # ok: both (B,S)
                pass
            else:
                # other mismatch
                if tuple(mask.shape) != tuple(logits.shape):
                    raise ValueError(
                        f"Shape mismatch: logits={tuple(logits.shape)} vs mask={tuple(mask.shape)}."
                    )

            if self.from_logits:
                probs = torch.sigmoid(logits) if logits.ndim == 2 else torch.sigmoid(logits)
            else:
                probs = logits  # assume already probabilities

            probs_f = probs.reshape(probs.shape[0], -1)
            tgt_f = mask.to(dtype=probs.dtype).reshape(mask.shape[0], -1)

            inter = (probs_f * tgt_f).sum(dim=1)
            denom = probs_f.sum(dim=1) + tgt_f.sum(dim=1)

            dice = (2.0 * inter + eps) / (denom + eps)  # (B,)
            loss_per_sample = 1.0 - dice                # (B,)

        # ------------------------
        # reduction across batch
        # ------------------------
        if self.reduction == "mean":
            return loss_per_sample.mean()
        elif self.reduction == "sum":
            return loss_per_sample.sum()
        else:
            raise RuntimeError(f"Unexpected reduction {self.reduction!r}")
        
class SoftDiceLoss(Objective):
    """
    Large-volume optimized Soft Dice for segmentation with NaN-missing masks.

    Assumptions:
      - targets are batched tensors and may contain NaNs for missing supervision.
      - For a given sample b:
          if all voxels are NaN => sample has no mask (skip if optional)
          else finite voxels encode class ids in [0, C-1] (or [0,1] for binary)

    Expected:
      logits:  (B,C,D,H,W) or (B,1,D,H,W) or (B,D,H,W)
      targets: (B,D,H,W) float or int, with NaNs for missing supervision
    """

    def __init__(
        self,
        logits_path: str,
        target_path: str,
        *,
        name: str = "soft_dice_loss",
        weight: float = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        is_optional: bool = True,
        include_background: bool = False,
        smooth: float = 1e-6,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(
            name=name,
            weight=weight,
            reduction=reduction,
            is_optional=is_optional,
        )
        self._logits_path = logits_path
        self._target_path = f"{target_path}"

        self.include_background = bool(include_background)
        self.smooth = float(smooth)
        self.eps = float(eps)

        self._required_keys = (self._logits_path, self._target_path)

    @property
    def required_keys(self) -> tuple[str, ...]:
        return self._required_keys

    @staticmethod
    def _as_two_class_probs_from_binary_logits(logits: Tensor) -> Tensor:
        """
        logits: (B,1,...) or (B,...)
        returns probs: (B,2,V)
        """
        if logits.ndim >= 2 and logits.shape[1] == 1:
            logit = logits[:, 0]  # (B, ...)
        else:
            logit = logits  # (B, ...)
        p1 = torch.sigmoid(logit)                 # (B, ...)
        p0 = (1.0 - p1)
        # flatten to V and stack into 2-class
        B = p1.shape[0]
        p1f = p1.reshape(B, -1)
        p0f = p0.reshape(B, -1)
        return torch.stack([p0f, p1f], dim=1)     # (B,2,V)

    def loss(self, *, inputs: dict[str, Tensor]) -> Tensor:
        logits = self.resolve(inputs, self._logits_path)
        targets = self.resolve(inputs, self._target_path)

        if targets.ndim != 4:
            raise ValueError(f"targets must be (B,D,H,W), got {tuple(targets.shape)}")

        # ---------------- decide valid samples (per-sample masks present) ----------------
        # missing supervision encoded as NaN over the whole volume for that sample
        finite = torch.isfinite(targets)                      # (B,D,H,W) bool
        has_mask = finite.view(finite.shape[0], -1).any(dim=1)  # (B,) bool

        if not bool(has_mask.any().item()):
            # no supervision in the entire batch
            if self.is_optional:
                return self._zero_loss(inputs)
            raise ValueError(
                f"Objective '{self.name}' received a batch with no valid masks (all-NaN targets). "
                "Mark the objective optional or ensure masks exist."
            )

        # filter to supervised samples only (keeps compute bounded)
        logits = logits[has_mask]
        targets = targets[has_mask]
        finite = finite[has_mask]

        B = targets.shape[0]
        V = targets.shape[1] * targets.shape[2] * targets.shape[3]

        # ---------------- build probs (flattened) ----------------
        # probs_flat: (B,C,V)
        if logits.ndim == 5:
            C = logits.shape[1]
            if C == 1:
                probs_flat = self._as_two_class_probs_from_binary_logits(logits)  # (B,2,V)
                C = 2
            else:
                probs = torch.softmax(logits, dim=1)            # (B,C,D,H,W)
                probs_flat = probs.reshape(B, C, -1)            # (B,C,V)
        elif logits.ndim == 4:
            # treat as binary logit (B,D,H,W)
            probs_flat = self._as_two_class_probs_from_binary_logits(logits)      # (B,2,V)
            C = 2
        else:
            raise ValueError(f"logits must be (B,C,D,H,W) or (B,D,H,W), got {tuple(logits.shape)}")

        # ---------------- flatten targets + valid voxels ----------------
        finite_flat = finite.reshape(B, -1)  # (B,V)

        # targets may be float (with NaNs); replace invalid with 0 before casting
        # (invalid voxels will be masked out anyway)
        t = targets.clone()
        t[~finite] = 0
        t_flat = t.reshape(B, -1).long()     # (B,V)

        # sanity clamp: prevent out-of-range labels from exploding scatter
        # (library misuse -> crash is ok; but this makes it safer)
        if (t_flat.min() < 0) or (t_flat.max() >= C):
            raise ValueError(
                f"Targets contain class ids outside [0, {C-1}]. "
                f"Observed min={int(t_flat.min())}, max={int(t_flat.max())}."
            )

        # apply voxel mask by zeroing contribution where invalid
        # we'll use weights w_flat in scatter_add / sums
        w_flat = finite_flat.to(probs_flat.dtype)  # (B,V) float {0,1}

        # ---------------- compute Dice components without one-hot ----------------
        # pred_sum[c] = Σ_v p(c|v) over valid voxels
        pred_sum = (probs_flat * w_flat.unsqueeze(1)).sum(dim=2)   # (B,C)

        # target_sum[c] = count of voxels with label c over valid voxels
        target_sum = torch.zeros((B, C), device=probs_flat.device, dtype=probs_flat.dtype)
        # scatter-add ones into class bins
        target_sum.scatter_add_(dim=1, index=t_flat, src=w_flat)   # (B,C)

        # intersection[c] = Σ_{v: y=v==c} p(c|v) over valid voxels
        # gather p(y|v): (B,V)
        p_true = probs_flat.gather(dim=1, index=t_flat.unsqueeze(1)).squeeze(1)  # (B,V)
        # mask invalid voxels
        p_true = p_true * w_flat

        inter = torch.zeros((B, C), device=probs_flat.device, dtype=probs_flat.dtype)
        inter.scatter_add_(dim=1, index=t_flat, src=p_true)  # (B,C)

        # ---------------- dice per class ----------------
        denom = pred_sum + target_sum
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth + self.eps)    # (B,C)

        # optionally drop background (class 0)
        if not self.include_background and C > 1:
            dice = dice[:, 1:]

        # dice loss = 1 - mean(dice)
        # average over classes then batch
        dice_mean = dice.mean()
        loss = 1.0 - dice_mean

        # reduction (scalar already, but keep contract)
        if self.reduction == "sum":
            # scalar; sum == identity here
            return loss
        return loss
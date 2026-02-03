# some Relational Objectives (evaluated on prediction-target)

"""
Implemented Relational Objectives:
 + CrossEntropyLoss
 + MSELoss
 + More to come...
"""

from sktorch.modules.nn.objectives._base import LossOut, RelationalObjective
import torch.nn.functional as F


class CrossEntropyLoss(RelationalObjective):
    """
    Cross-entropy classification loss.

    A relational objective that computes the cross-entropy loss between
    predicted logits and target class labels.

    This objective compares model outputs directly against ground-truth
    targets and is typically used for multi-class classification tasks.

    ### Call signature:
    ```
    obj(predictions, targets, context=None) -> LossOut
    ```

    ### Parameters:
    - **name** (`str`, default=`"cross_entropy_loss"`):
      Human-readable identifier for the objective. Used for logging,
      diagnostics, and composite objective naming.

    - **weight** (`float`, default=`1.0`):
      Scalar weight applied to this loss when used inside a composite objective.

    - **required** (`bool`, default=`True`):
      If `True`, missing required inputs raise immediately.
      If `False`, the objective may be skipped and replaced with a zero-loss
      (only if a graph-connected zero-loss can be constructed).

    - **reduction** (`str`, default=`"mean"`):
      Specifies the reduction to apply to the output:
        - `"mean"`: average loss over the batch (recommended)
        - `"sum"`: sum loss over the batch
        
      Note: `"none"` is not supported here because this framework requires
      objectives to return a scalar loss tensor.

    ### Required keys:
    - **predictions** must contain:
        - `"clf/logits"` — raw, unnormalized class logits.
    - **targets** must contain:
        - `"clf/targets"` — integer class labels.

    ### Example:
    ```python
    ce_loss = CrossEntropyLoss(reduction="mean")

    out = ce_loss(
        predictions={"clf/logits": logits},
        targets={"clf/targets": labels},
    )

    scalar_loss = out.loss
    ```
    """

    def __init__(
        self,
        name: str = "cross_entropy_loss",
        required: bool = True,
        weight: float = 1.0,
        *,
        reduction: str = "mean",
    ):
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"{self.__class__.__name__} reduction must be one of ('mean','sum'), got {reduction!r}."
            )

        super().__init__(
            name=name,
            required=required,
            weight=weight,
            required_pred_keys=("clf/logits",),
            required_target_keys=("clf/targets",),
        )
        self._reduction = reduction

    def loss(self, predictions, targets):
        """
        Compute the cross-entropy loss.

        ### Inputs:
        - **predictions["clf/logits"]** (`Tensor`):
          Logits of shape `(N, C)` where `C` is the number of classes.
        - **targets["clf/targets"]** (`Tensor`):
          Integer class labels of shape `(N,)`.

        ### Returns:
        - `LossOut` containing a scalar loss tensor (per `reduction`).
        """
        logits = predictions["clf/logits"]
        labels = targets["clf/targets"]

        loss = F.cross_entropy(logits, labels, reduction=self._reduction)
        return LossOut(loss=loss, details={"reduction": self._reduction})


class MSELoss(RelationalObjective):
    """
    Mean squared error regression loss.

    A relational objective that computes the mean squared error (MSE)
    between predicted values and regression targets.

    This objective is commonly used for regression tasks where model outputs
    are expected to approximate continuous target values.

    ### Call signature:
    ```
    obj(predictions, targets, context=None) -> LossOut
    ```

    ### Parameters:
    - **name** (`str`, default=`"mse_loss"`):
      Human-readable identifier for the objective.

    - **weight** (`float`, default=`1.0`):
      Scalar weight applied to this loss when used inside a composite objective.

    - **required** (`bool`, default=`True`):
      If `True`, missing required inputs raise immediately.
      If `False`, the objective may be skipped and replaced with a zero-loss
      (only if a graph-connected zero-loss can be constructed).

    - **reduction** (`str`, default=`"mean"`):
      Specifies the reduction to apply to the output:
        - `"mean"`: average loss over the batch (recommended)
        - `"sum"`: sum loss over the batch

      Note: `"none"` is not supported here because this framework requires
      objectives to return a scalar loss tensor.

    ### Required keys:
    - **predictions** must contain:
        - `"reg/pred"` — predicted regression outputs.
    - **targets** must contain:
        - `"reg/target"` — ground-truth regression targets.

    ### Example:
    ```python
    mse_loss = MSELoss(reduction="mean")

    out = mse_loss(
        predictions={"reg/pred": preds},
        targets={"reg/target": targets},
    )

    scalar_loss = out.loss
    ```
    """

    def __init__(
        self,
        name: str = "mse_loss",
        required: bool = True,
        weight: float = 1.0,
        *,
        reduction: str = "mean",
    ):
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"{self.__class__.__name__} reduction must be one of ('mean','sum'), got {reduction!r}."
            )

        super().__init__(
            name=name,
            required=required,
            weight=weight,
            required_pred_keys=("reg/pred",),
            required_target_keys=("reg/target",),
        )
        self._reduction = reduction

    def loss(self, predictions, targets):
        """
        Compute the mean squared error loss.

        ### Inputs:
        - **predictions["reg/pred"]** (`Tensor`):
          Predicted values of shape `(N, ...)`.
        - **targets["reg/target"]** (`Tensor`):
          Target values of matching shape.

        ### Returns:
        - `LossOut` containing a scalar loss tensor (per `reduction`).
        """
        preds = predictions["reg/pred"]
        tgts = targets["reg/target"]

        loss = F.mse_loss(preds, tgts, reduction=self._reduction)
        return LossOut(loss=loss, details={"reduction": self._reduction})

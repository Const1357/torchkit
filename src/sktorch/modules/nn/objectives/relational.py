# some Relational Objectives (evaluated on prediction-target)

"""
Implemented Relational Objectives:
 + CrossEntropyLoss
 + MSELoss
 + More to come...
"""

from __future__ import annotations

from typing import Any, Mapping

import torch.nn.functional as F
from torch import Tensor

from sktorch.modules.nn.objectives._base import LossOut, RelationalObjective


class CrossEntropyLoss(RelationalObjective):
    """
    Cross-entropy classification loss.

    A relational objective that computes the cross-entropy loss between
    predicted logits and target class labels.

    ### Call signature:
    ```
    obj(predictions, targets, context=None) -> LossOut
    ```

    Parameters
    ----------
    task : str, default="clf"
        Task identifier used to namespace prediction and target keys.
        Required keys are constructed as:
            - predictions[f"{task}/logits"]
            - targets[f"{task}/targets"]

    name : str, default="cross_entropy_loss"
        Objective name used for logging and diagnostics.

    required : bool, default=True
        If True, missing required inputs raise immediately.
        If False, the objective may be skipped and replaced with a zero-loss
        (only if a zero-loss can be constructed under the base-objective rules).

    weight : float, default=1.0
        Scalar weight applied by composite objectives.

    reduction : {"mean", "sum"}, default="mean"
        Reduction applied to the per-sample cross-entropy loss.

        Note: "none" is intentionally not supported. By convention, objectives
        in this framework return a scalar loss tensor. Unreduced diagnostics
        should be returned via LossOut.details.

    Required keys
    -------------
    predictions:
        - f"{task}/logits": raw, unnormalized logits of shape (N, C)
    targets:
        - f"{task}/targets": integer class labels of shape (N,)

    Example
    -------
    ```python
    ce = CrossEntropyLoss(task="clf", reduction="mean")

    out = ce(
        predictions={"clf/logits": logits},
        targets={"clf/targets": labels},
    )

    loss = out.loss
    ```
    """

    def __init__(
        self,
        *,
        task: str = "clf",
        name: str = "cross_entropy_loss",
        required: bool = True,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        if not isinstance(task, str) or not task:
            raise ValueError(f"{self.__class__.__name__} task must be a non-empty string, got {task!r}.")
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"{self.__class__.__name__} reduction must be one of ('mean','sum'), got {reduction!r}."
            )

        self._task = task
        self._reduction = reduction

        super().__init__(
            name=name,
            required=required,
            weight=weight,
            required_pred_keys=(f"{task}/logits",),
            required_target_keys=(f"{task}/targets",),
        )

    def loss(
        self,
        predictions: Mapping[str, Tensor | None],
        targets: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> LossOut:
        """
        Compute the cross-entropy loss.

        Returns
        -------
        LossOut
            Scalar loss tensor according to the configured reduction.
        """
        logits = predictions[f"{self._task}/logits"]
        labels = targets[f"{self._task}/targets"]

        # Base-objective contract guarantees required keys are present and non-None
        assert isinstance(logits, Tensor)
        assert isinstance(labels, Tensor)

        loss = F.cross_entropy(logits, labels, reduction=self._reduction)
        return LossOut(loss=loss, details={"reduction": self._reduction})


class MSELoss(RelationalObjective):
    """
    Mean squared error regression loss.

    A relational objective that computes the mean squared error (MSE) between
    predicted values and regression targets.

    ### Call signature:
    ```
    obj(predictions, targets, context=None) -> LossOut
    ```

    Parameters
    ----------
    task : str, default="reg"
        Task identifier used to namespace prediction and target keys.
        Required keys are constructed as:
            - predictions[f"{task}/pred"]
            - targets[f"{task}/target"]

    name : str, default="mse_loss"
        Objective name used for logging and diagnostics.

    required : bool, default=True
        If True, missing required inputs raise immediately.
        If False, the objective may be skipped and replaced with a zero-loss
        (only if a zero-loss can be constructed under the base-objective rules).

    weight : float, default=1.0
        Scalar weight applied by composite objectives.

    reduction : {"mean", "sum"}, default="mean"
        Reduction applied to the per-element squared error.

        Note: "none" is intentionally not supported. By convention, objectives
        in this framework return a scalar loss tensor. Unreduced diagnostics
        should be returned via LossOut.details.

    Required keys
    -------------
    predictions:
        - f"{task}/pred": predicted values of shape (N, ...)
    targets:
        - f"{task}/target": target values of matching shape

    Example
    -------
    ```python
    mse = MSELoss(task="reg", reduction="mean")

    out = mse(
        predictions={"reg/pred": preds},
        targets={"reg/target": tgts},
    )

    loss = out.loss
    ```
    """

    def __init__(
        self,
        *,
        task: str = "reg",
        name: str = "mse_loss",
        required: bool = True,
        weight: float = 1.0,
        reduction: str = "mean",
    ):
        if not isinstance(task, str) or not task:
            raise ValueError(f"{self.__class__.__name__} task must be a non-empty string, got {task!r}.")
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"{self.__class__.__name__} reduction must be one of ('mean','sum'), got {reduction!r}."
            )

        self._task = task
        self._reduction = reduction

        super().__init__(
            name=name,
            required=required,
            weight=weight,
            required_pred_keys=(f"{task}/pred",),
            required_target_keys=(f"{task}/target",),
        )

    def loss(
        self,
        predictions: Mapping[str, Tensor | None],
        targets: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> LossOut:
        """
        Compute the mean squared error loss.

        Returns
        -------
        LossOut
            Scalar loss tensor according to the configured reduction.
        """
        preds = predictions[f"{self._task}/pred"]
        tgts = targets[f"{self._task}/target"]

        # Base-objective contract guarantees required keys are present and non-None
        assert isinstance(preds, Tensor)
        assert isinstance(tgts, Tensor)

        loss = F.mse_loss(preds, tgts, reduction=self._reduction)
        return LossOut(loss=loss, details={"reduction": self._reduction})

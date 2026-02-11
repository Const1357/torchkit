# some Intrinsic Objectives (evaluated on predictions only)

"""
Implemented Intrinsic Objectives:
 + EntropyTerm
 + More to come...
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from torch import Tensor

from sktorch.modules.nn.objectives._base import LossOut, IntrinsicObjective
from sktorch.modules.nn.objectives._mixins.signed_intrinsic import SignedIntrinsicObjectiveMixin


class EntropyTerm(SignedIntrinsicObjectiveMixin, IntrinsicObjective):
    """
    Entropy regularization term.

    An intrinsic objective that computes the entropy of predicted class
    probabilities. This loss depends only on model predictions and does
    not require external targets.

    The optimization direction controls the effect:
    - "maximize" (default): encourages higher entropy (uncertainty / exploration).
    - "minimize": encourages lower entropy (confidence / sharp distributions).

    Internally, this objective always returns a scalar loss to be minimized.
    When direction="maximize", the entropy is negated.

    ### Call signature:
    ```
    obj(predictions, context=None) -> LossOut
    ```

    Parameters
    ----------
    task : str, default="clf"
        Task identifier used to namespace prediction keys.
        Required key is constructed as:
            - predictions[f"{task}/probs"]

    name : str, default="entropy_term"
        Objective name used for logging and diagnostics.

    required : bool, default=True
        If True, missing required inputs raise immediately.
        If False, the objective may be skipped and replaced with a zero-loss
        (only if a zero-loss can be constructed under the base-objective rules).

    weight : float, default=1.0
        Scalar weight applied by composite objectives.

    direction : {"maximize", "minimize"}, default="maximize"
        Optimization direction for entropy:
            - "maximize": maximize entropy (implemented as minimizing negative entropy).
            - "minimize": minimize entropy directly.

    Required keys
    -------------
    predictions:
        - f"{task}/probs": class probabilities (after softmax), shape (N, C)

    Example
    -------
    ```python
    # encourage exploration / uncertainty
    entropy_term = EntropyTerm(task="clf", weight=0.01, direction="maximize")

    out = entropy_term(
        predictions={"clf/probs": probs},
    )

    loss = out.loss
    ```
    """

    def __init__(
        self,
        *,
        task: str = "clf",
        name: str = "entropy_term",
        required: bool = True,
        weight: float = 1.0,
        direction: Literal["maximize", "minimize"] = "maximize",
    ):
        if not isinstance(task, str) or not task:
            raise ValueError(f"{self.__class__.__name__} task must be a non-empty string, got {task!r}.")
        if direction not in ("maximize", "minimize"):
            raise ValueError(
                f"{self.__class__.__name__} direction must be one of "
                f"('maximize','minimize'), got {direction!r}."
            )

        self._task = task

        IntrinsicObjective.__init__(
            self,
            name=name,
            required=required,
            weight=weight,
            required_pred_keys=(f"{task}/probs",),
        )
        SignedIntrinsicObjectiveMixin.__init__(self, direction=direction)

    def loss(
        self,
        predictions: Mapping[str, Tensor | None],
        context: Mapping[str, Any] | None = None,
    ) -> LossOut:
        """
        Compute the entropy-based loss according to the chosen direction.

        Returns
        -------
        LossOut
            Scalar loss tensor to be minimized.
        """
        probs = predictions[f"{self._task}/probs"]

        # Base-objective contract guarantees required keys are present and non-None
        assert isinstance(probs, Tensor)

        probs = probs.clamp_min(1e-10)
        entropy = -(probs * probs.log()).sum(dim=1).mean()  # scalar

        loss = self._apply_direction(entropy)  # maximize => minimize -entropy
        return LossOut(loss=loss, details=self._with_direction_details({}))

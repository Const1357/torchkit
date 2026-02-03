# some Intrinsic Objectives (evaluated on predictions only)

"""
Implemented Intrinsic Objectives:
 + EntropyTerm
 + More to come...
"""

import torch    # for Tensor operations
from typing import Literal
from sktorch.modules.nn.objectives._base import LossOut, IntrinsicObjective


class EntropyTerm(IntrinsicObjective):
    """
    Entropy regularization term.

    An intrinsic objective that computes the entropy of predicted class
    probabilities. This loss depends only on model predictions and does
    not require external targets.

    The optimization **direction** controls the effect:
    - `"maximize"` (default): encourages higher entropy (uncertainty,
      smoother distributions, exploration).
    - `"minimize"`: encourages lower entropy (confidence, sharper
      distributions).

    Internally, this objective always returns a scalar loss to be
    **minimized**. When `direction="maximize"`, the entropy is negated.

    ### Call signature:
    ```
    obj(predictions, context=None) -> LossOut
    ```

    ### Parameters:
    - **name** (`str`, default=`"entropy_term"`):
      Human-readable identifier for the objective. Used for logging,
      diagnostics, and composite objective naming.

    - **weight** (`float`, default=`1.0`):
      Scalar weight applied to this term when used inside a composite
      objective.

    - **required** (`bool`, default=`True`):
      If `True`, missing required inputs raise immediately.
      If `False`, the objective may be skipped and replaced with a zero-loss
      (only if a graph-connected zero-loss can be constructed).

    - **direction** (`{"maximize", "minimize"}`, default=`"maximize"`):

      Optimization direction for entropy:
        - `"maximize"`: maximize entropy (implemented as minimizing **negative** entropy).
        - `"minimize"`: minimize entropy directly.

    ### Required keys:
    - **predictions** must contain:
        - `"clf/probs"` — class probabilities (after softmax).

    ### Example:
    ```python
    # encourage exploration / uncertainty
    entropy_term = EntropyTerm(weight=0.01, direction="maximize")

    # encourage confident predictions
    confidence_term = EntropyTerm(weight=0.01, direction="minimize")

    out = entropy_term(
        predictions={"clf/probs": probs},
    )

    scalar_loss = out.loss
    ```
    """

    def __init__(
        self,
        name: str = "entropy_term",
        required: bool = True,
        weight: float = 1.0,
        *,
        direction: Literal["maximize", "minimize"] = "maximize",
    ):
        if direction not in ("maximize", "minimize"):
            raise ValueError(
                f"{self.__class__.__name__} direction must be one of "
                f"('maximize','minimize'), got {direction!r}."
            )

        super().__init__(
            name=name,
            required=required,
            weight=weight,
            required_pred_keys=("clf/probs",),
        )
        self._direction = direction

    def loss(self, predictions):
        """
        Compute the entropy-based loss according to the chosen direction.

        ### Inputs:
        - **predictions["clf/probs"]** (`Tensor`):
          Probability tensor of shape `(N, C)` where `C` is the number of classes.
          Values are expected to sum to 1 along the class dimension.

        ### Returns:
        - `LossOut` containing a scalar loss tensor to be **minimized**.
        """
        probs = predictions["clf/probs"].clamp_min(1e-10)
        entropy = -(probs * probs.log()).sum(dim=1).mean()

        # framework minimizes losses:
        # - maximize entropy -> minimize negative entropy
        # - minimize entropy -> minimize entropy directly
        loss = -entropy if self._direction == "maximize" else entropy

        return LossOut(
            loss=loss,
            details={"direction": self._direction},
        )

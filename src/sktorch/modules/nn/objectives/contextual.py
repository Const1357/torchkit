# some Contextual Objectives (evaluated primarily on context)

"""
Implemented Contextual Objectives:
 + L2Penalty
 + More to come...
"""

from __future__ import annotations

from typing import Any, Mapping, Iterable
from collections.abc import Iterator

from sktorch.modules.nn.objectives._base import LossOut, ContextualObjective
import torch
from torch import Tensor


def _iter_tensors(obj: Any) -> Iterator[Tensor]:
    """
    Yield Tensors found inside `obj`.

    Supported containers:
    - Tensor
    - nn.Parameter (is a Tensor)
    - iterables (list/tuple/set/generator)
    - mappings (dict-like)
    """
    if obj is None:
        return
    if isinstance(obj, Tensor):
        yield obj
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            yield from _iter_tensors(v)
        return
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            yield from _iter_tensors(v)
        return
    # generic iterable (e.g. model.parameters())
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        for v in obj:
            yield from _iter_tensors(v)
        return
    # everything else: ignore


class L2Penalty(ContextualObjective):
    """
    L2 penalty (weight decay) objective.

    A contextual objective that computes an L2 penalty over a set of parameters
    provided via `context`. Predictions and targets are not required.

    Typical use cases include weight decay, parameter norm penalties,
    or constraint-based regularization.

    ### Call signature:
    ```
    obj(context, predictions=None, targets=None) -> LossOut
    ```

    ### Parameters:
    - **name** (`str`, default=`"l2_penalty"`):  
      Human-readable identifier for the objective. Used for logging,
      diagnostics, and composite objective naming.

    - **weight** (`float`, default=`1.0`):  
      Scalar weight applied to this term when used inside a composite
      objective.

    - **required** (`bool`, default=`True`):  
      If `True`, missing required inputs raise immediately.  
      If `False`, the objective may be skipped and replaced with a zero-loss
      (only if a graph-connected zero-loss can be constructed).

    - **include_bias** (`bool`, default=`False`):  
      If `False`, parameters that look like biases (1D tensors) are excluded.

    ### Required keys:
    - **context** must contain:
        - `"params"` — an iterable (or nested container) of tensors/parameters.

    ### Example:
    ```python
    l2 = L2Penalty(weight=1e-4)

    out = l2(
        context={"params": model.parameters()},
    )

    scalar_loss = out.loss
    ```
    """

    def __init__(
        self,
        name: str = "l2_penalty",
        required: bool = True,
        weight: float = 1.0,
        *,
        include_bias: bool = False,
    ):
        super().__init__(
            name=name,
            required=required,
            weight=weight,
            required_context_keys=("params",),
            required_pred_keys=(),
            required_target_keys=(),
        )
        self._include_bias = bool(include_bias)

    def loss(
        self,
        context: Mapping[str, Any],
        predictions: Mapping[str, Tensor | None] | None = None,
        targets: Mapping[str, Tensor | None] | None = None,
    ) -> LossOut:
        """
        Compute the L2 penalty over parameters provided in `context`.

        ### Inputs:
        - **context["params"]**:  
          Iterable (or nested container) of tensors/parameters to penalize.

        ### Returns:
        - `LossOut` containing a scalar L2 penalty loss.
        """
        params_obj = context["params"]

        total_sq: Tensor | None = None
        n_params: int = 0

        for p in _iter_tensors(params_obj):
            # optionally skip "bias-like" parameters (common convention)
            if (not self._include_bias) and p.ndim == 1:
                continue

            # sum of squares
            s = (p * p).sum()
            total_sq = s if total_sq is None else (total_sq + s)
            n_params += 1

        if total_sq is None:
            raise ValueError(
                f"{self.__class__.__name__} expected at least one parameter tensor in context['params'], "
                f"but found none (after filtering)."
            )

        return LossOut(
            loss=total_sq,
            details={
                "num_tensors": n_params,
                "include_bias": self._include_bias,
            },
        )

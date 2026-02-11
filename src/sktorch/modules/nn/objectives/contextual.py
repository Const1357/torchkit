# some Contextual Objectives (evaluated primarily on context)

"""
Implemented Contextual Objectives:
 + L2Penalty
 + More to come...
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Iterable, Mapping

from torch import Tensor

from sktorch.modules.nn.objectives._base import ContextualObjective, LossOut


def _iter_tensors(obj: Any) -> Iterator[Tensor]:
    """
    Yield Tensors found inside `obj`.

    Supported containers
    --------------------
    - Tensor (including nn.Parameter)
    - mappings (dict-like)
    - iterables (list/tuple/set/generator)
    - generic iterables (e.g., model.parameters())

    Notes
    -----
    Non-tensor values are ignored. This helper is intentionally permissive to
    support nested parameter containers.
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

    ### Call signature:
    ```
    obj(context, predictions=None, targets=None) -> LossOut
    ```

    Parameters
    ----------
    name : str, default="l2_penalty"
        Objective name used for logging and diagnostics.
    required : bool, default=True
        If True, missing required inputs raise immediately.
        If False, the objective may be skipped and replaced with a zero-loss
        (only if a zero-loss can be constructed under the base-objective rules).
    weight : float, default=1.0
        Scalar weight applied by composite objectives.
    key : str, default="params"
        Context key that contains an iterable (or nested container) of tensors/parameters.
    include_bias : bool, default=False
        If False, parameters that look like biases (1D tensors) are excluded.

    Required keys
    -------------
    context:
        - `key` (default: "params") must be present and non-None.

    Example
    -------
    ```python
    l2 = L2Penalty(weight=1e-4)

    out = l2(
        context={"params": model.parameters()},
    )

    loss = out.loss
    ```
    """

    def __init__(
        self,
        name: str = "l2_penalty",
        required: bool = True,
        weight: float = 1.0,
        *,
        key: str = "params",
        include_bias: bool = False,
    ):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{self.__class__.__name__} key must be a non-empty string, got {key!r}.")

        self._key = key
        self._include_bias = bool(include_bias)

        super().__init__(
            name=name,
            required=required,
            weight=weight,
            required_context_keys=(key,),
            required_pred_keys=(),
            required_target_keys=(),
        )

    def loss(
        self,
        context: Mapping[str, Any],
        predictions: Mapping[str, Tensor | None] | None = None,
        targets: Mapping[str, Tensor | None] | None = None,
    ) -> LossOut:
        """
        Compute the L2 penalty over parameters provided in `context[key]`.

        Returns
        -------
        LossOut
            Scalar L2 penalty loss and diagnostics in details.
        """
        params_obj = context[self._key]

        total_sq: Tensor | None = None
        n_tensors: int = 0

        for p in _iter_tensors(params_obj):
            # optionally skip "bias-like" parameters (common convention)
            if (not self._include_bias) and p.ndim == 1:
                continue

            s = (p * p).sum()
            total_sq = s if total_sq is None else (total_sq + s)
            n_tensors += 1

        if total_sq is None:
            raise ValueError(
                f"{self.__class__.__name__} expected at least one parameter tensor in context[{self._key!r}], "
                f"but found none (after filtering)."
            )

        return LossOut(
            loss=total_sq,
            details={
                "num_tensors": n_tensors,
                "include_bias": self._include_bias,
                "key": self._key,
            },
        )

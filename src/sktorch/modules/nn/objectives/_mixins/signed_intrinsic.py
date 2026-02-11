from __future__ import annotations
from typing import Literal


from typing import Literal, Any, Dict
from torch import Tensor


class SignedIntrinsicObjectiveMixin:
    """
    Mixin for IntrinsicObjective subclasses that want a `direction` option.

    Semantics:
    - The framework always minimizes `LossOut.loss`.
    - If direction == "minimize": loss_to_minimize = value
    - If direction == "maximize": loss_to_minimize = -value

    Usage:
        class EntropyTerm(SignedIntrinsicObjectiveMixin, IntrinsicObjective):
            def __init__(..., direction="maximize", ...):
                IntrinsicObjective.__init__(...)
                SignedIntrinsicObjectiveMixin.__init__(direction=direction)

            def loss(...):
                entropy = ...
                loss = self._apply_direction(entropy)
                return LossOut(loss=loss, details=self._with_direction_details({}))
    """

    _direction: Literal["minimize", "maximize"]

    def __init__(self, *, direction: Literal["minimize", "maximize"] = "minimize"):
        if direction not in ("minimize", "maximize"):
            raise ValueError(
                f"{self.__class__.__name__} direction must be one of "
                f"('minimize','maximize'), got {direction!r}."
            )
        self._direction = direction

    @property
    def direction(self) -> Literal["minimize", "maximize"]:
        return self._direction

    def _apply_direction(self, value: Tensor) -> Tensor:
        """
        Convert an objective value into a loss-to-minimize according to direction.
        """
        if value.ndim != 0:
            raise ValueError(
                f"{self.__class__.__name__} expected a scalar Tensor to sign, got shape {tuple(value.shape)}."
            )
        return value if self._direction == "minimize" else -value

    def _with_direction_details(self, details: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Convenience helper: returns a dict that includes the direction.
        """
        d: Dict[str, Any] = {} if details is None else dict(details)
        d.setdefault("direction", self._direction)
        return d
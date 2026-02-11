from __future__ import annotations

from sktorch.modules.nn.objectives._base import (
    _BaseObjective,
    RelationalObjective,
    IntrinsicObjective,
    ContextualObjective,
    LossOut,
)

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override

from typing import Any, Dict, Mapping, TypeAlias

from torch import Tensor


class CompositeObjective(_BaseObjective):
    """
    Interface class for composite objectives.

    A composite objective aggregates multiple objectives into a single
    weighted loss. Each contained objective is evaluated using the same
    inputs and combined according to its individual weight.

    Nested composite objectives are supported.

    ### Call signature:
    ```
    obj(predictions=None, targets=None, context=None) -> LossOut
    ```

    ### Parameters:
    - `*objectives`: A variable number of objectives to combine. Each objective must be
      an instance of `IntrinsicObjective`, `RelationalObjective`, `ContextualObjective`, or
      `CompositeObjective`. Alternatively, a single list or tuple of objectives can be provided.
    - `required` (bool): If `True`, at least one of the contained objectives must be required.
      (Otherwise, the composite itself being required would be meaningless.)

    Notes:
    - This composite reduces by weighted sum and therefore requires each sub-objective
      to return a scalar loss tensor (ndim == 0).
    """

    def __init__(
        self,
        *objectives: "Objective" | list["Objective"] | tuple["Objective", ...],
        required: bool = True,
    ):
        if len(objectives) == 1 and isinstance(objectives[0], (list, tuple)):
            objs = list(objectives[0])
        else:
            objs = list(objectives)

        if not objs:
            raise ValueError(f"At least one objective must be provided. Got {len(objs)}.")

        for i, obj in enumerate(objs):
            if not isinstance(obj, _BaseObjective):
                raise TypeError(
                    f"All objectives must be derived from _BaseObjective, got type {type(obj)} at index {i}."
                )

        if required and all(obj.required is False for obj in objs):
            raise ValueError("At least one objective must be required if the composite objective is required.")

        self._objectives: tuple[_BaseObjective, ...] = tuple(objs)

        # New base objective signature: no `objective_type`.
        # Composite weight is conceptually 1.0; contained objectives carry their own weights.
        super().__init__(
            name=f"Mix:{'+'.join(f'{obj.weight}*{obj.name}' for obj in self._objectives)}",
            weight=1.0,
            required=required,
            required_pred_keys=(),
            required_target_keys=(),
            required_context_keys=(),
        )

    @property
    def objectives(self) -> tuple[_BaseObjective, ...]:
        return self._objectives

    @property
    def num_objectives(self) -> int:
        return len(self._objectives)

    @property
    def objective_names(self) -> tuple[str, ...]:
        return tuple(obj.name for obj in self._objectives)

    @property
    def union_pred_keys(self) -> tuple[str, ...]:
        return tuple(sorted(set().union(*(obj.required_pred_keys for obj in self._objectives))))

    @property
    def union_target_keys(self) -> tuple[str, ...]:
        return tuple(sorted(set().union(*(obj.required_target_keys for obj in self._objectives))))

    @property
    def union_context_keys(self) -> tuple[str, ...]:
        return tuple(sorted(set().union(*(obj.required_context_keys for obj in self._objectives))))

    def _call_objective(
        self,
        obj: _BaseObjective,
        *,
        predictions: Mapping[str, Tensor | None] | None,
        targets: Mapping[str, Tensor | None] | None,
        context: Mapping[str, Any] | None,
    ) -> LossOut:
        """
        Call an objective with signature-correct arguments.

        Important: the latest base objective contract expects required keys to be present and non-None.
        For convenience, if predictions/targets/context are None, we pass empty mappings so that:
        - required objectives raise (missing keys),
        - optional objectives skip (if a zero-loss can be constructed).
        """
        preds = predictions if predictions is not None else {}
        targs = targets if targets is not None else {}
        ctx = context if context is not None else {}

        if isinstance(obj, CompositeObjective):
            return obj(predictions=preds, targets=targs, context=ctx)
        if isinstance(obj, RelationalObjective):
            return obj(predictions=preds, targets=targs, context=ctx)
        if isinstance(obj, IntrinsicObjective):
            return obj(predictions=preds, context=ctx)
        if isinstance(obj, ContextualObjective):
            return obj(context=ctx, predictions=preds, targets=targs)

        raise TypeError(f"CompositeObjective {self.name} - Unsupported objective type: {type(obj)}")

    @override
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"objectives=[{', '.join(repr(obj) for obj in self._objectives)}], "
            f"required={self._required}"
            f")"
        )

    @override
    def __call__(
        self,
        *,
        predictions: Mapping[str, Tensor | None] | None = None,
        targets: Mapping[str, Tensor | None] | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> LossOut:
        """Compute the composite loss as weighted sum of individual objectives."""

        total: Tensor | None = None
        details: Dict[str, Any] = {}

        for obj in self._objectives:
            obj_loss_out = self._call_objective(
                obj,
                predictions=predictions,
                targets=targets,
                context=context,
            )

            l = obj_loss_out.loss
            d = obj_loss_out.details

            # Composite defines reduction policy: each objective must produce a scalar loss.
            if l.ndim != 0:
                raise ValueError(
                    f"Objective loss must be a scalar tensor (shape [ ]), got shape {tuple(l.shape)} "
                    f"from objective {obj.name}."
                )
            if not isinstance(d, dict):
                raise TypeError(f"Objective '{obj.name}' expects details of type dict, got {type(d)}.")

            weighted_loss = obj.weight * l
            total = weighted_loss if total is None else total + weighted_loss

            # Detach to avoid retaining graphs. Logging decides cpu/scalars.
            details[f"{obj.name}/loss"] = l.detach()
            details[f"{obj.name}/weighted_loss"] = weighted_loss.detach()

            # Flatten details dict.
            for key, value in d.items():
                details[f"{obj.name}/{key}"] = value.detach() if isinstance(value, Tensor) else value

        assert total is not None  # sanity
        return LossOut(loss=total, details=details)


Objective: TypeAlias = IntrinsicObjective | RelationalObjective | ContextualObjective | CompositeObjective

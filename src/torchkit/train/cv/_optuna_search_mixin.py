from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TypeAlias

import optuna


SuggestionType = Literal[
    "categorical",
    "float",
    "int",
    "loguniform",
    "uniform",
    "discrete_uniform",
]

TrialStatus = Literal[
    "SUCCESS",
    "FAILED",
    "PRUNED",
]

DerivedTransformFn: TypeAlias = Callable[..., Any]


@dataclass(frozen=True)
class SuggestionSpec:
    """
    One primitive Optuna suggestion.

    `values` follows the existing library contract:
    - categorical: `[choice0, choice1, ...]`
    - float/uniform/loguniform/int: `[low, high]`
    - discrete_uniform: `[low, high, step]`
    """

    values: list[Any]
    suggestion_type: SuggestionType
    projection_map: dict[Any, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, list):
            raise TypeError(
                f"SuggestionSpec.values must be a list, got {type(self.values).__name__}."
            )
        if len(self.values) == 0:
            raise ValueError("SuggestionSpec.values must be non-empty.")
        if self.projection_map is not None and not isinstance(self.projection_map, dict):
            raise TypeError(
                f"SuggestionSpec.projection_map must be a dict or None, got {type(self.projection_map).__name__}."
            )
        if self.projection_map is not None:
            missing = [value for value in self.values if value not in self.projection_map]
            if missing:
                raise ValueError(
                    "SuggestionSpec.projection_map must contain entries for all categorical choices. "
                    f"Missing: {missing}."
                )


@dataclass(frozen=True)
class DerivedParam:
    """
    A derived flattened parameter assignment computed from one or more sampled suggestions.

    Example:
        DerivedParam(
            target_path="model/heads/clf/head_module/input_dim",
            args=["model/backbone/encoder_channels"],
            transform=lambda channels: channels[-1],
        )

    `args` names keys from `ParameterGrid.suggestions`. `transform` receives those sampled
    values positionally in the order given by `args`.

    Note:
    `transform` may be any callable, including a lambda. Anonymous callables are often not
    pickleable, so if you need to pickle raw parameter-grid metadata, prefer top-level
    named functions.
    """

    target_path: str
    args: list[str]
    transform: DerivedTransformFn

    def __post_init__(self) -> None:
        if not isinstance(self.target_path, str) or not self.target_path:
            raise ValueError(
                f"DerivedParam.target_path must be a non-empty string, got {self.target_path!r}."
            )
        if not isinstance(self.args, list) or not self.args:
            raise ValueError("DerivedParam.args must be a non-empty list[str].")
        for arg in self.args:
            if not isinstance(arg, str) or not arg:
                raise ValueError(f"DerivedParam.args must contain non-empty strings, got {arg!r}.")
        if not callable(self.transform):
            raise TypeError(
                f"DerivedParam.transform must be callable, got {type(self.transform).__name__}."
            )


@dataclass
class ParameterGrid:
    """
    Structured parameter grid consumed by Optuna CV searchers.

    `suggestions` holds the primitive Optuna suggestions keyed by their flattened destination
    paths, usually `model/...` or `trainer/...`.

    `derived_params` holds optional additional flattened assignments computed after suggestion.
    Derived params can depend on one or more suggestion keys via `args`.

    Example:
        ParameterGrid(
            suggestions={
                "model/backbone/encoder_channels": SuggestionSpec(
                    values=[(32, 64, 128, 256), (64, 128, 256, 512)],
                    suggestion_type="categorical",
                ),
            },
            derived_params=[
                DerivedParam(
                    target_path="model/heads/clf/head_module/input_dim",
                    args=["model/backbone/encoder_channels"],
                    transform=lambda channels: channels[-1],
                ),
            ],
        )
    """

    suggestions: dict[str, SuggestionSpec]
    derived_params: list[DerivedParam] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.suggestions, dict) or not self.suggestions:
            raise ValueError("ParameterGrid.suggestions must be a non-empty dict[str, SuggestionSpec].")

        normalized: dict[str, SuggestionSpec] = {}
        for key, spec in self.suggestions.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"Suggestion key must be a non-empty string, got {key!r}.")
            normalized[key] = coerce_suggestion_spec(key, spec)
        self.suggestions = normalized

        if not isinstance(self.derived_params, list):
            raise TypeError(
                f"ParameterGrid.derived_params must be a list[DerivedParam], got {type(self.derived_params).__name__}."
            )
        for derived in self.derived_params:
            if not isinstance(derived, DerivedParam):
                raise TypeError(
                    f"ParameterGrid.derived_params must contain DerivedParam instances, got {type(derived).__name__}."
                )
            missing_args = [arg for arg in derived.args if arg not in self.suggestions]
            if missing_args:
                raise ValueError(
                    f"DerivedParam for target {derived.target_path!r} references unknown suggestion keys: {missing_args}."
                )

        derived_targets = [derived.target_path for derived in self.derived_params]
        if len(derived_targets) != len(set(derived_targets)):
            raise ValueError("ParameterGrid.derived_params contains duplicate target_path values.")

    @classmethod
    def from_simple(
        cls,
        suggestions: dict[str, SuggestionSpec | tuple[list[Any], SuggestionType]],
    ) -> "ParameterGrid":
        return cls(suggestions=suggestions)


SimpleSuggestionEntry: TypeAlias = (
    SuggestionSpec
    | tuple[list[Any], SuggestionType]
    | tuple[list[Any], SuggestionType, dict[Any, Any]]
)
ParameterGridLike: TypeAlias = ParameterGrid | dict[str, SimpleSuggestionEntry]


def coerce_suggestion_spec(
    param_name: str,
    spec: SimpleSuggestionEntry,
) -> SuggestionSpec:
    if isinstance(spec, SuggestionSpec):
        return spec

    if not isinstance(spec, tuple):
        raise TypeError(
            f"Suggestion spec for {param_name!r} must be SuggestionSpec or tuple[list[Any], SuggestionType], "
            f"got {type(spec).__name__}."
        )
    if len(spec) not in (2, 3):
        raise ValueError(
            f"Tuple suggestion spec for {param_name!r} must have length 2 or 3, got {len(spec)}."
        )

    if len(spec) == 2:
        values, suggestion_type = spec
        projection_map = None
    else:
        values, suggestion_type, projection_map = spec
    return SuggestionSpec(
        values=values,
        suggestion_type=suggestion_type,
        projection_map=projection_map,
    )


def coerce_parameter_grid(parameter_grid: ParameterGridLike) -> ParameterGrid:
    if isinstance(parameter_grid, ParameterGrid):
        return parameter_grid
    if isinstance(parameter_grid, dict):
        return ParameterGrid.from_simple(parameter_grid)
    raise TypeError(
        f"parameter_grid must be a ParameterGrid or dict[str, suggestion_spec], got {type(parameter_grid).__name__}."
    )


class OptunaSearchMixin:
    """
    Optuna-specific mixin for search-based CV runners.

    Searchers consume a `ParameterGrid`:
    - `suggestions` are sampled directly by Optuna.
    - `derived_params` are computed after sampling and flattened into additional concrete
      `model/...` or `trainer/...` assignments.

    The final output of `suggest_parameters(...)` is always a flat dict[path, value] that can
    be routed into model/trainer specs without extra logic in the CV train loops.
    """

    @staticmethod
    def suggest_parameters(
        trial: optuna.Trial,
        parameter_grid: ParameterGridLike,
    ) -> dict[str, Any]:
        grid = coerce_parameter_grid(parameter_grid)

        sampled_values: dict[str, Any] = {}
        for param_name, spec in grid.suggestions.items():
            param_values = spec.values
            suggestion_type = spec.suggestion_type

            if suggestion_type == "categorical":
                sampled_value = trial.suggest_categorical(param_name, param_values)
                if spec.projection_map is not None:
                    sampled_value = spec.projection_map[sampled_value]

            elif suggestion_type == "float":
                sampled_value = trial.suggest_float(param_name, *param_values)

            elif suggestion_type == "int":
                sampled_value = trial.suggest_int(param_name, *param_values)

            elif suggestion_type == "loguniform":
                sampled_value = trial.suggest_float(param_name, *param_values, log=True)

            elif suggestion_type == "uniform":
                sampled_value = trial.suggest_float(param_name, *param_values)

            elif suggestion_type == "discrete_uniform":
                low, high, q = param_values
                sampled_value = trial.suggest_float(param_name, low, high, step=q)

            else:
                raise ValueError(
                    f"Unsupported suggestion_type {suggestion_type!r} for parameter {param_name!r}."
                )

            sampled_values[param_name] = sampled_value

        flattened_params = dict(sampled_values)
        for derived in grid.derived_params:
            if derived.target_path in flattened_params:
                raise ValueError(
                    f"Derived parameter target_path {derived.target_path!r} collides with an existing sampled key."
                )

            arg_values = [sampled_values[arg] for arg in derived.args]
            flattened_params[derived.target_path] = derived.transform(*arg_values)

        return flattened_params

    def _create_study(self) -> optuna.Study:
        """
        Selection scores are always maximized. Raw metrics are converted into
        selection scores via BaseCV._to_selection_score(...).
        """
        return optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=2,
                n_warmup_steps=5,
                interval_steps=1,
            ),
        )

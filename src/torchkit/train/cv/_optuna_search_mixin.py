from __future__ import annotations

from typing import Any, Literal, Tuple

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


class OptunaSearchMixin:
    """
    Optuna-specific mixin for search-based CV runners.

    Assumes the concrete class uses a parameter grid with the contract:
        dict[str, tuple[list, SuggestionType]]
    """

    @staticmethod
    def suggest_parameters(
        trial: optuna.Trial,
        parameter_grid: dict[str, Tuple[list, SuggestionType]],
    ) -> dict[str, Any]:
        suggested_params: dict[str, Any] = {}

        for param_name, (param_values, suggestion_type) in parameter_grid.items():
            if suggestion_type == "categorical":
                suggested_params[param_name] = trial.suggest_categorical(param_name, param_values)

            elif suggestion_type == "float":
                suggested_params[param_name] = trial.suggest_float(param_name, *param_values)

            elif suggestion_type == "int":
                suggested_params[param_name] = trial.suggest_int(param_name, *param_values)

            elif suggestion_type == "loguniform":
                suggested_params[param_name] = trial.suggest_float(param_name, *param_values, log=True)

            elif suggestion_type == "uniform":
                suggested_params[param_name] = trial.suggest_float(param_name, *param_values)

            elif suggestion_type == "discrete_uniform":
                low, high, q = param_values
                suggested_params[param_name] = trial.suggest_float(param_name, low, high, step=q)

            else:
                raise ValueError(
                    f"Unsupported suggestion_type {suggestion_type!r} for parameter {param_name!r}."
                )

        return suggested_params

    def _create_study(self) -> optuna.Study:
        """
        Selection scores are always maximized. Raw metrics are converted into
        selection scores via BaseCV._to_selection_score(...).
        """
        return optuna.create_study(direction="maximize")
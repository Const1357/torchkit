from __future__ import annotations

import pytest

from torchkit.train.cv._optuna_search_mixin import (
    DerivedParam,
    OptunaSearchMixin,
    ParameterGrid,
    SuggestionSpec,
    coerce_parameter_grid,
)


class _FakeTrial:
    def suggest_categorical(self, name, values):
        del name
        return values[-1]

    def suggest_float(self, name, low, high, log=False, step=None):
        del name, high, log
        if step is not None:
            return low + step
        return low

    def suggest_int(self, name, low, high):
        del name, high
        return low


def test_parameter_grid_from_simple_coerces_tuple_entries():
    grid = coerce_parameter_grid(
        {
            "trainer/config/max_epochs": ([10, 20], "categorical"),
            "trainer/config/optimizer_kwargs/lr": ([1e-5, 1e-3], "loguniform"),
        }
    )

    assert isinstance(grid, ParameterGrid)
    assert isinstance(grid.suggestions["trainer/config/max_epochs"], SuggestionSpec)
    assert grid.suggestions["trainer/config/max_epochs"].values == [10, 20]
    assert grid.suggestions["trainer/config/max_epochs"].suggestion_type == "categorical"


def test_parameter_grid_supports_multi_argument_derived_mapping():
    grid = ParameterGrid(
        suggestions={
            "trainer/config/optimizer_kwargs/solver": SuggestionSpec(
                values=["sgd", "adamw"],
                suggestion_type="categorical",
            ),
            "trainer/config/optimizer_kwargs/base_lr": SuggestionSpec(
                values=[1e-4, 1e-2],
                suggestion_type="loguniform",
            ),
        },
        derived_params=[
            DerivedParam(
                target_path="trainer/config/optimizer_kwargs/effective_lr",
                args=[
                    "trainer/config/optimizer_kwargs/solver",
                    "trainer/config/optimizer_kwargs/base_lr",
                ],
                transform=lambda solver, lr: lr * 0.1 if solver == "sgd" else lr,
            ),
        ],
    )

    params = OptunaSearchMixin.suggest_parameters(_FakeTrial(), grid)

    assert params["trainer/config/optimizer_kwargs/solver"] == "adamw"
    assert params["trainer/config/optimizer_kwargs/base_lr"] == pytest.approx(1e-4)
    assert params["trainer/config/optimizer_kwargs/effective_lr"] == pytest.approx(1e-4)


def test_parameter_grid_rejects_unknown_derived_args():
    with pytest.raises(ValueError, match="unknown suggestion keys"):
        ParameterGrid(
            suggestions={
                "trainer/config/max_epochs": SuggestionSpec(
                    values=[10, 20],
                    suggestion_type="categorical",
                ),
            },
            derived_params=[
                DerivedParam(
                    target_path="trainer/config/early_stopping_patience",
                    args=["trainer/config/does_not_exist"],
                    transform=lambda x: x,
                ),
            ],
        )


def test_parameter_grid_rejects_derived_target_collision_with_sampled_key():
    grid = ParameterGrid(
        suggestions={
            "trainer/config/max_epochs": SuggestionSpec(
                values=[10, 20],
                suggestion_type="categorical",
            ),
        },
        derived_params=[
            DerivedParam(
                target_path="trainer/config/max_epochs",
                args=["trainer/config/max_epochs"],
                transform=lambda x: x + 1,
            ),
        ],
    )

    with pytest.raises(ValueError, match="collides with an existing sampled key"):
        OptunaSearchMixin.suggest_parameters(_FakeTrial(), grid)

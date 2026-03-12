from __future__ import annotations

import json
import pickle

import optuna
import pandas as pd
import pytest
import torch
from torch.utils.data import Subset

from torchkit.data.split import StratifiedKFold
from torchkit.evaluate.classification_evaluator import ClassificationEvaluator
from torchkit.train.cv._base_cv import (
    BaseCV,
    _safe_take,
    _clone_tensor_dict,
    _clone_state_dict_cpu,
    _concat_tensor_dicts,
    _resolve_original_indices_for_subset,
)
from torchkit.train.cv._base_search_cv import BaseSearchCV
from torchkit.train.cv._optuna_results import (
    FoldResult,
    OptunaTrialResult,
    OptunaSearchCVResult,
    OuterFoldResult,
    NestedOptunaSearchCVResult,
)
from torchkit.train.cv._optuna_search_mixin import OptunaSearchMixin
from torchkit.train.factory import TrainerFactory

from tests.torchkit.test_cv_and_runners.conftest import (
    ErrorRateEvaluator,
    make_model_spec,
    make_trainer_spec,
    make_optuna_search_cv,
)


class MinimalBaseCV(BaseCV):
    pass


class MinimalBaseSearchCV(BaseSearchCV):
    pass


class MinimalOptunaSearchCV(OptunaSearchMixin, BaseSearchCV):
    pass


def test_safe_take_supports_list_and_tensor():
    values_list = ["a", "b", "c", "d"]
    values_tensor = torch.tensor([10, 20, 30, 40])
    idx = [0, 2]

    assert _safe_take(values_list, idx) == ["a", "c"]
    assert torch.equal(_safe_take(values_tensor, idx), torch.tensor([10, 30]))


def test_clone_tensor_dict_and_state_dict_cpu():
    x = {"a": torch.randn(2, 3), "b": torch.randn(1)}
    cloned = _clone_tensor_dict(x)

    assert cloned.keys() == x.keys()
    for k in x:
        assert torch.equal(cloned[k], x[k].cpu())
        assert cloned[k].device.type == "cpu"
        assert cloned[k] is not x[k]

    state = {"w": torch.randn(4, 2)}
    cloned_state = _clone_state_dict_cpu(state)
    assert cloned_state is not None
    assert torch.equal(cloned_state["w"], state["w"].cpu())
    assert cloned_state["w"] is not state["w"]

    assert _clone_state_dict_cpu(None) is None


def test_concat_tensor_dicts_concatenates_per_key():
    d1 = {"clf": torch.tensor([[1.0, 2.0]])}
    d2 = {"clf": torch.tensor([[3.0, 4.0]])}
    out = _concat_tensor_dicts([d1, d2])

    assert "clf" in out
    assert out["clf"].shape == (2, 2)
    assert torch.equal(out["clf"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_resolve_original_indices_for_nested_subset(tiny_dataset):
    s1 = Subset(tiny_dataset, [2, 5, 8, 11])
    s2 = Subset(s1, [1, 3])

    resolved = _resolve_original_indices_for_subset(s2)
    assert resolved == [5, 11]


def test_base_cv_selection_metric_helpers_maximize_and_minimize(tmp_path):
    model_spec = make_model_spec()
    max_trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        )
    )
    min_trainer_spec = make_trainer_spec(
        evaluator=ErrorRateEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
        )
    )

    max_cv = MinimalBaseCV(
        model_spec=model_spec,
        trainer_spec=max_trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        k_outer=2,
        final_model_dir=str(tmp_path / "max"),
    )
    min_cv = MinimalBaseCV(
        model_spec=model_spec,
        trainer_spec=min_trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        k_outer=2,
        final_model_dir=str(tmp_path / "min"),
    )

    assert max_cv._selection_metric_name() == "accuracy"
    assert max_cv._selection_metric_direction() == "maximize"
    assert max_cv._to_selection_score(0.8) == pytest.approx(0.8)

    assert min_cv._selection_metric_name() == "error_rate"
    assert min_cv._selection_metric_direction() == "minimize"
    assert min_cv._to_selection_score(0.2) == pytest.approx(-0.2)


def test_base_cv_rejects_unrebuildable_configuration():
    model_spec = make_model_spec()
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        )
    )

    with pytest.raises(ValueError, match="unrebuildable"):
        MinimalBaseCV(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            outer_splitter_cls=StratifiedKFold,
            k_outer=2,
            final_model_dir=None,
            keep_final_model_state_dict_cpu=False,
        )


def test_base_search_cv_routes_parameters_into_real_specs(tmp_path):
    model_spec = make_model_spec()
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    cv = MinimalBaseSearchCV(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        parameter_grid={
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([4], "categorical"),
        },
        outer_splitter_cls=StratifiedKFold,
        k_outer=2,
        final_model_dir=str(tmp_path),
    )

    built_model_spec, built_trainer_spec, trainer = cv._build_trainer_for_trial(
        params={
            "model/backbone/kwargs/scale_factor": -1.0,
            "trainer/config/max_epochs": 7,
        }
    )

    assert built_model_spec.backbone.kwargs["scale_factor"] == -1.0
    assert built_trainer_spec.config.max_epochs == 7
    assert trainer.config.max_epochs == 7


def test_optuna_search_mixin_suggest_parameters():
    study = optuna.create_study(direction="maximize")
    trial = study.ask()

    params = MinimalOptunaSearchCV.suggest_parameters(
        trial,
        {
            "a": ([1, 2], "categorical"),
            "b": ([0.1, 0.5], "float"),
            "c": ([1, 3], "int"),
            "d": ([1e-4, 1e-2], "loguniform"),
            "e": ([0.0, 1.0], "uniform"),
            "f": ([0.0, 1.0, 0.25], "discrete_uniform"),
        },
    )

    assert params["a"] in [1, 2]
    assert 0.1 <= params["b"] <= 0.5
    assert 1 <= params["c"] <= 3
    assert 1e-4 <= params["d"] <= 1e-2
    assert 0.0 <= params["e"] <= 1.0
    assert params["f"] in [0.0, 0.25, 0.5, 0.75, 1.0]


def test_results_containers_support_offline_processing_and_reconstruction(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    y, _groups = tiny_labels_groups

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    cv = make_optuna_search_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=None,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    # Core result APIs
    d = result.to_dict()
    assert isinstance(d, dict)
    assert d["best_trial_number"] == 0

    j = result.to_json()
    parsed = json.loads(j)
    assert parsed["best_trial_number"] == 0

    trials_df = result.trials_to_dataframe()
    folds_df = result.folds_to_dataframe()
    lb_row = result.leaderboard_row()

    assert isinstance(trials_df, pd.DataFrame)
    assert isinstance(folds_df, pd.DataFrame)
    assert isinstance(lb_row, dict)
    assert len(trials_df) == 1
    assert len(folds_df) == 2
    assert lb_row["best_trial_number"] == 0

    # Reconstruction from live result
    rebuilt_model = result.rebuild_final_model(device="cpu")
    rebuilt_trainer = result.rebuild_final_trainer(device="cpu")
    assert rebuilt_model is not None
    assert rebuilt_trainer is not None

    # Pickle roundtrip preserves reconstructibility
    blob = pickle.dumps(result)
    restored: OptunaSearchCVResult = pickle.loads(blob)

    restored_model = restored.rebuild_final_model(device="cpu")
    restored_trainer = restored.rebuild_final_trainer(device="cpu")
    assert restored_model is not None
    assert restored_trainer is not None

    # Selected-trial helpers
    selected_trial = restored.selected_trial_result()
    assert isinstance(selected_trial, OptunaTrialResult)
    assert selected_trial.trial_number == restored.best_trial_number

    selected_trial_folds_df = selected_trial.folds_to_dataframe()
    assert isinstance(selected_trial_folds_df, pd.DataFrame)
    assert len(selected_trial_folds_df) == 2


def test_nested_results_support_offline_processing_and_reconstruction(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    from torchkit.train.cv.nested_optuna_search_cv import NestedOptunaSearchCV

    y, groups = tiny_labels_groups

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ClassificationEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
            primary_metric="accuracy",
        ),
        max_epochs=2,
    )

    from .conftest import make_nested_cv
    cv = make_nested_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        k_outer=2,
        k_inner=2,
        random_state=None,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    d = result.to_dict()
    assert isinstance(d, dict)
    assert len(d["outer_results"]) == 2

    j = result.to_json()
    parsed = json.loads(j)
    assert len(parsed["outer_results"]) == 2

    outer_df = result.outer_folds_to_dataframe()
    lb_row = result.leaderboard_row()

    assert isinstance(outer_df, pd.DataFrame)
    assert len(outer_df) == 2
    assert isinstance(lb_row, dict)

    rebuilt_model = result.rebuild_final_model(0, device="cpu")
    rebuilt_trainer = result.rebuild_final_trainer(0, device="cpu")
    assert rebuilt_model is not None
    assert rebuilt_trainer is not None

    blob = pickle.dumps(result)
    restored: NestedOptunaSearchCVResult = pickle.loads(blob)

    restored_model = restored.rebuild_final_model(0, device="cpu")
    restored_trainer = restored.rebuild_final_trainer(0, device="cpu")
    assert restored_model is not None
    assert restored_trainer is not None

    outer0 = restored.outer_results[0]
    assert isinstance(outer0, OuterFoldResult)
    assert isinstance(outer0.inner_search_result, OptunaSearchCVResult)
    assert outer0.outer_test_metrics == outer0.inner_search_result.holdout_metrics
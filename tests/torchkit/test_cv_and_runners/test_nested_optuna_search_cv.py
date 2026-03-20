from __future__ import annotations

import os
import pickle

import pytest
import torch

from torchkit.data.split import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from torchkit.evaluate.select import AccuracySelectorEvaluator
from torchkit.models.Model.factory import TorchkitModelFactory

from torchkit.train.cv._optuna_results import (
    NestedOptunaSearchCVResult,
    OptunaSearchCVResult,
    OuterFoldResult,
)

from tests.torchkit.test_cv_and_runners.conftest import (
    ErrorRateEvaluator,
    make_model_spec,
    make_trainer_spec,
    make_nested_cv,
)


def test_nested_cv_rejects_unrebuildable_final_model_configuration():
    from torchkit.train.cv.nested_optuna_search_cv import NestedOptunaSearchCV

    model_spec = make_model_spec()
    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        )
    )

    with pytest.raises(ValueError, match="unrebuildable"):
        NestedOptunaSearchCV(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
            outer_splitter_cls=StratifiedKFold,
            inner_splitter_cls=StratifiedKFold,
            n_trials=1,
            k_outer=2,
            k_inner=2,
            final_model_dir=None,
            keep_final_model_state_dict_cpu=False,
        )


def test_nested_cv_logs_everything_needed_for_reporting_stratified(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    y, _groups = tiny_labels_groups

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=2,
    )

    cv = make_nested_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        parameter_grid={
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([2], "categorical"),
        },
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        k_outer=2,
        k_inner=2,
        random_state=None,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)
    assert isinstance(result, NestedOptunaSearchCVResult)

    assert result.outer_splitter_name == "StratifiedKFold"
    assert result.inner_splitter_name == "StratifiedKFold"
    assert result.k_outer == 2
    assert result.k_inner == 2
    assert result.n_trials == 1
    assert result.max_trial_attempts == 3
    assert result.selection_metric_name == "dataset:classification"
    assert result.selection_metric_direction == "maximize"
    assert result.final_model_dir == str(tmp_path)

    assert len(result.outer_results) == 2
    all_dataset_indices = set(range(len(tiny_dataset)))

    for outer in result.outer_results:
        assert isinstance(outer, OuterFoldResult)
        assert isinstance(outer.inner_search_result, OptunaSearchCVResult)

        assert len(outer.outer_train_indices) > 0
        assert len(outer.outer_test_indices) > 0
        assert set(outer.outer_train_indices).isdisjoint(set(outer.outer_test_indices))
        assert set(outer.outer_train_indices) | set(outer.outer_test_indices) == all_dataset_indices

        inner = outer.inner_search_result

        assert inner.attempted_trials == 1
        assert inner.successful_trials == 1
        assert inner.failed_trials == 0
        assert inner.pruned_trials == 0
        assert len(inner.trial_results) == 1

        assert outer.best_trial_number == 0
        assert outer.best_params["model/backbone/kwargs/scale_factor"] == 1.0
        assert outer.best_params["trainer/config/max_epochs"] == 2
        assert outer.best_metric == pytest.approx(1.0)
        assert outer.best_selection_score == pytest.approx(1.0)

        # Inner search result should have exact OOF coverage of outer-train pool
        trial = inner.selected_trial_result()
        assert sorted(trial.aggregate_oof_sample_indices) == sorted(inner.search_pool_indices)
        assert sorted(inner.search_pool_indices) == sorted(outer.outer_train_indices)
        assert set(trial.aggregate_oof_sample_indices).isdisjoint(set(outer.outer_test_indices))

        # Outer holdout must be stored both places by current design
        assert outer.outer_test_metrics is not None
        assert inner.holdout_metrics is not None
        assert outer.outer_test_metrics == inner.holdout_metrics
        assert "val/classification" in outer.outer_test_metrics
        assert outer.outer_test_metrics["val/classification"] == pytest.approx(1.0)

        # Final saved model should exist inside outer-fold subdir
        assert inner.final_model_state_dict_path is not None
        assert os.path.exists(inner.final_model_state_dict_path)


@pytest.mark.parametrize(
    "outer_splitter_cls,inner_splitter_cls",
    [
        (GroupKFold, GroupKFold),
        (StratifiedGroupKFold, StratifiedGroupKFold),
    ],
)
def test_nested_cv_group_splitters_have_no_group_leakage(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
    outer_splitter_cls,
    inner_splitter_cls,
):
    y, groups = tiny_labels_groups

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=2,
    )

    cv = make_nested_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        outer_splitter_cls=outer_splitter_cls,
        inner_splitter_cls=inner_splitter_cls,
        parameter_grid={"model/backbone/kwargs/scale_factor": ([1.0], "categorical")},
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        k_outer=2,
        k_inner=2,
        random_state=None,
    )

    result = cv.run(tiny_dataset, index=y, groups=groups)

    for outer in result.outer_results:
        outer_train_groups = {groups[i] for i in outer.outer_train_indices}
        outer_test_groups = {groups[i] for i in outer.outer_test_indices}
        assert outer_train_groups.isdisjoint(outer_test_groups)

        trial = outer.inner_search_result.selected_trial_result()
        for fold in trial.fold_results:
            fold_train_groups = {groups[i] for i in fold.train_indices}
            fold_val_groups = {groups[i] for i in fold.val_indices}
            assert fold_train_groups.isdisjoint(fold_val_groups)


def test_nested_cv_handles_minimize_selection_metric_correctly(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    y, _groups = tiny_labels_groups

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=ErrorRateEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
        ),
        max_epochs=2,
    )

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

    assert result.selection_metric_name == "dataset:error_rate"
    assert result.selection_metric_direction == "maximize"

    for outer in result.outer_results:
        assert outer.best_metric == pytest.approx(0.0)
        assert outer.best_selection_score == pytest.approx(0.0)
        assert outer.outer_test_metrics is not None
        assert "val/error_rate" in outer.outer_test_metrics
        assert outer.outer_test_metrics["val/error_rate"] == pytest.approx(0.0)


def test_nested_cv_rebuilds_final_model_and_trainer_and_preserves_calibrator_fit(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    y, _groups = tiny_labels_groups

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=2,
    )

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
        keep_final_model_state_dict_cpu=False,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    rebuilt_model = result.rebuild_final_model(0, device="cpu")
    phead = rebuilt_model.prediction_heads["clf"]
    calibrator = phead.calibrator
    assert calibrator is not None
    assert int(calibrator.fit_calls.item()) == 1

    outer0 = result.outer_results[0]
    assert int(calibrator.last_num_samples.item()) == len(outer0.outer_train_indices)
    assert int(calibrator.last_num_classes.item()) == 2

    sample = tiny_dataset[0]
    batched_sample = {
        "x": sample["x"].unsqueeze(0),
        "y": sample["y"].unsqueeze(0),
    }
    pred = rebuilt_model.predict(
        batched_sample,
        "clf",
        return_raw_head_outputs=True,
    )
    assert "clf" in pred
    assert "logits" in pred["clf"]
    assert "calibrated_logits" in pred["clf"]

    rebuilt_trainer = result.rebuild_final_trainer(0, device="cpu")
    assert rebuilt_trainer is not None


def test_nested_cv_result_is_pickleable_and_reconstruction_survives_roundtrip(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    y, _groups = tiny_labels_groups

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=2,
    )

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

    blob = pickle.dumps(result)
    restored: NestedOptunaSearchCVResult = pickle.loads(blob)

    model = restored.rebuild_final_model(0, device="cpu")
    trainer = restored.rebuild_final_trainer(0, device="cpu")
    assert model is not None
    assert trainer is not None

    sample = tiny_dataset[0]
    batched_sample = {
        "x": sample["x"].unsqueeze(0),
        "y": sample["y"].unsqueeze(0),
    }
    pred = model.predict(
        batched_sample,
        "clf",
        return_raw_head_outputs=True,
    )
    assert "clf" in pred
    assert "logits" in pred["clf"]
    assert "calibrated_logits" in pred["clf"]


def test_nested_cv_final_model_reconstruction_is_prediction_identical(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    y, _groups = tiny_labels_groups

    model_spec = make_model_spec(scale_factor=1.0)
    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=2,
    )

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
        keep_final_model_state_dict_cpu=True,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)
    inner_result = result.outer_results[0].inner_search_result

    assert inner_result.final_model_spec is not None
    assert inner_result.final_model_state_dict_cpu is not None
    assert inner_result.final_model_state_dict_path is not None

    rebuilt_default = result.rebuild_final_model(0, device="cpu")
    rebuilt_from_cpu = TorchkitModelFactory.build(
        inner_result.final_model_spec,
        state_dict=inner_result.final_model_state_dict_cpu,
        device="cpu",
    )
    rebuilt_from_path = TorchkitModelFactory.build(
        inner_result.final_model_spec,
        state_dict_path=inner_result.final_model_state_dict_path,
        device="cpu",
    )

    sample = tiny_dataset[0]
    batched_sample = {
        "x": sample["x"].unsqueeze(0),
        "y": sample["y"].unsqueeze(0),
    }

    pred_default = rebuilt_default.predict(
        batched_sample,
        "clf",
        return_raw_head_outputs=True,
    )
    pred_cpu = rebuilt_from_cpu.predict(
        batched_sample,
        "clf",
        return_raw_head_outputs=True,
    )
    pred_path = rebuilt_from_path.predict(
        batched_sample,
        "clf",
        return_raw_head_outputs=True,
    )

    assert torch.equal(pred_default["clf"]["logits"], pred_cpu["clf"]["logits"])
    assert torch.equal(pred_default["clf"]["logits"], pred_path["clf"]["logits"])
    assert torch.equal(
        pred_default["clf"]["calibrated_logits"],
        pred_cpu["clf"]["calibrated_logits"],
    )
    assert torch.equal(
        pred_default["clf"]["calibrated_logits"],
        pred_path["clf"]["calibrated_logits"],
    )


def test_nested_cv_is_deterministic_for_same_seed(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    y, groups = tiny_labels_groups

    model_spec_1 = make_model_spec(scale_factor=1.0)
    trainer_spec_1 = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=2,
    )

    model_spec_2 = make_model_spec(scale_factor=1.0)
    trainer_spec_2 = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=2,
    )

    cv1 = make_nested_cv(
        model_spec=model_spec_1,
        trainer_spec=trainer_spec_1,
        outer_splitter_cls=StratifiedGroupKFold,
        inner_splitter_cls=StratifiedGroupKFold,
        parameter_grid={
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([2], "categorical"),
        },
        tmp_path=tmp_path / "run1",
        n_trials=1,
        max_trial_attempts=3,
        k_outer=2,
        k_inner=2,
        random_state=42,
    )

    cv2 = make_nested_cv(
        model_spec=model_spec_2,
        trainer_spec=trainer_spec_2,
        outer_splitter_cls=StratifiedGroupKFold,
        inner_splitter_cls=StratifiedGroupKFold,
        parameter_grid={
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([2], "categorical"),
        },
        tmp_path=tmp_path / "run2",
        n_trials=1,
        max_trial_attempts=3,
        k_outer=2,
        k_inner=2,
        random_state=42,
    )

    result1 = cv1.run(tiny_dataset, index=y, groups=groups)
    result2 = cv2.run(tiny_dataset, index=y, groups=groups)

    assert len(result1.outer_results) == len(result2.outer_results)

    for outer1, outer2 in zip(result1.outer_results, result2.outer_results):
        assert outer1.outer_train_indices == outer2.outer_train_indices
        assert outer1.outer_test_indices == outer2.outer_test_indices

        inner1 = outer1.inner_search_result
        inner2 = outer2.inner_search_result

        assert inner1.best_trial_number == inner2.best_trial_number
        assert inner1.best_params == inner2.best_params
        assert inner1.best_metric == pytest.approx(inner2.best_metric)
        assert inner1.best_selection_score == pytest.approx(inner2.best_selection_score)

        assert inner1.selected_metric_mean == pytest.approx(inner2.selected_metric_mean)
        assert inner1.selected_metric_std == pytest.approx(inner2.selected_metric_std)
        assert inner1.selected_metric_min == pytest.approx(inner2.selected_metric_min)
        assert inner1.selected_metric_max == pytest.approx(inner2.selected_metric_max)

        assert inner1.holdout_metrics == inner2.holdout_metrics

        tr1 = inner1.selected_trial_result()
        tr2 = inner2.selected_trial_result()
        assert tr1.aggregate_oof_sample_indices == tr2.aggregate_oof_sample_indices
        for task in tr1.aggregate_oof_logits:
            assert torch.equal(tr1.aggregate_oof_logits[task], tr2.aggregate_oof_logits[task])
            assert torch.equal(tr1.aggregate_oof_targets[task], tr2.aggregate_oof_targets[task])

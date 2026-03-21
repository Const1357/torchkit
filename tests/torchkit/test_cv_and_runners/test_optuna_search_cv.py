from __future__ import annotations

import os
import pickle

import pytest
import torch

import optuna

from torchkit.data.split import StratifiedKFold, GroupKFold, StratifiedGroupKFold
from torchkit.evaluate.select import AccuracySelectorEvaluator
from torchkit.models.Model.factory import TorchkitModelFactory

from torchkit.train.cv._optuna_search_mixin import ParameterGrid
from torchkit.train.cv.optuna_search_cv import OptunaSearchCV
from torchkit.train.cv._optuna_results import OptunaSearchCVResult

from tests.torchkit.test_cv_and_runners.conftest import (
    ErrorRateEvaluator,
    make_model_spec,
    make_trainer_spec,
    make_optuna_search_cv,
)


def test_optuna_search_cv_rejects_unrebuildable_final_model_configuration():
    model_spec = make_model_spec()
    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        )
    )

    with pytest.raises(ValueError, match="unrebuildable"):
        OptunaSearchCV(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
            splitter_cls=StratifiedKFold,
            n_trials=1,
            n_splits=2,
            final_model_dir=None,
            keep_final_model_state_dict_cpu=False,
        )


def test_optuna_search_cv_rejects_invalid_parameter_path(tmp_path):
    model_spec = make_model_spec()
    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        )
    )

    with pytest.raises(ValueError, match="must start with 'model/' or 'trainer/'"):
        make_optuna_search_cv(
            model_spec=model_spec,
            trainer_spec=trainer_spec,
            splitter_cls=StratifiedKFold,
            parameter_grid=ParameterGrid.from_simple({"badpath": ([1.0], "categorical")}),
            tmp_path=tmp_path,
        )


def test_optuna_search_cv_logs_everything_needed_for_reporting_stratified(
    tiny_dataset,
    tiny_labels_groups,
    tiny_report_evaluator,
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

    cv = make_optuna_search_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([2], "categorical"),
        }),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=None,
        report_evaluator=tiny_report_evaluator,
    )

    result = cv.run(tiny_dataset, index=y, groups=None, holdout_dataset=tiny_dataset)

    assert isinstance(result, OptunaSearchCVResult)

    # CV-level metadata
    assert result.splitter_name == "StratifiedKFold"
    assert result.n_splits == 2
    assert result.n_trials == 1
    assert result.max_trial_attempts == 3
    assert result.selection_metric_name == "dataset:classification"
    assert result.selection_metric_direction == "maximize"
    assert result.final_model_dir == str(tmp_path)
    assert result.keep_final_model_state_dict_cpu is True

    # Search pool membership
    assert sorted(result.search_pool_indices) == list(range(len(tiny_dataset)))

    # Trial bookkeeping
    assert result.attempted_trials == 1
    assert result.successful_trials == 1
    assert result.failed_trials == 0
    assert result.pruned_trials == 0
    assert len(result.trial_results) == 1

    # Best-trial reporting
    assert result.best_trial_number == 0
    assert result.best_params["model/backbone/kwargs/scale_factor"] == 1.0
    assert result.best_params["trainer/config/max_epochs"] == 2
    assert result.best_metric == pytest.approx(1.0)
    assert result.best_selection_score == pytest.approx(1.0)

    # Selected-fold reporting
    assert len(result.selected_fold_results) == 2
    assert result.selected_metric_mean == pytest.approx(1.0)
    assert result.selected_metric_std == pytest.approx(0.0)
    assert result.selected_metric_min == pytest.approx(1.0)
    assert result.selected_metric_max == pytest.approx(1.0)

    # Final refit artifacts
    assert result.final_model_spec is not None
    assert result.final_trainer_spec is not None
    assert result.final_fit_epochs is not None
    assert result.final_fit_epochs >= 0
    assert result.final_epochs_ran is not None
    assert result.final_epochs_ran >= 0

    assert result.final_best_epoch is None
    assert result.final_best_metric is None

    assert len(result.final_train_logs) >= 1
    assert result.final_val_logs == []
    assert isinstance(result.final_history, list)
    assert len(result.final_history) == 1

    # Final saved model
    assert result.final_model_state_dict_cpu is not None
    assert result.final_model_state_dict_path is not None
    assert os.path.exists(result.final_model_state_dict_path)

    assert result.holdout_metrics is not None
    assert result.holdout_report_results is not None
    assert result.holdout_report_results["clf/accuracy"] == pytest.approx(1.0)
    assert result.holdout_report_results["clf/n_samples"] == len(tiny_dataset)
    assert isinstance(result.holdout_report_results["positive_logit_mean"], float)
    assert isinstance(result.holdout_report_results["batch_pred_labels"], list)

    # Successful trial details
    trial = result.selected_trial_result()
    assert trial.params["model/backbone/kwargs/scale_factor"] == 1.0
    assert trial.params["trainer/config/max_epochs"] == 2
    assert trial.aggregate_metric == pytest.approx(1.0)
    assert trial.aggregate_selection_score == pytest.approx(1.0)
    assert trial.error_message is None
    assert trial.error_traceback is None

    # OOF aggregate exact coverage of search pool, no duplicates
    assert sorted(trial.aggregate_oof_sample_indices) == sorted(result.search_pool_indices)
    assert len(trial.aggregate_oof_sample_indices) == len(set(trial.aggregate_oof_sample_indices))

    assert "clf" in trial.aggregate_oof_logits
    assert "clf" in trial.aggregate_oof_targets
    assert trial.aggregate_oof_logits["clf"].shape[0] == len(result.search_pool_indices)
    assert trial.aggregate_oof_targets["clf"].shape[0] == len(result.search_pool_indices)

    # Each fold
    seen_val_indices: set[int] = set()
    for fold in trial.fold_results:
        assert set(fold.train_indices).isdisjoint(set(fold.val_indices))
        assert sorted(fold.oof_sample_indices) == sorted(fold.val_indices)

        assert "clf" in fold.oof_logits
        assert "clf" in fold.oof_targets
        assert fold.oof_logits["clf"].shape[0] == len(fold.val_indices)
        assert fold.oof_targets["clf"].shape[0] == len(fold.val_indices)

        seen_val_indices.update(fold.val_indices)

    assert seen_val_indices == set(result.search_pool_indices)


def test_optuna_search_cv_stores_holdout_report_results(
    tiny_dataset,
    tiny_labels_groups,
    tiny_report_evaluator,
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

    cv = make_optuna_search_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        report_evaluator=tiny_report_evaluator,
    )

    result = cv.run(tiny_dataset, index=y, groups=None, holdout_dataset=tiny_dataset)

    assert result.report_evaluator is not None
    assert result.holdout_report_results is not None
    assert result.holdout_report_results["clf/accuracy"] == pytest.approx(1.0)
    assert result.holdout_report_results["clf/n_samples"] == len(tiny_dataset)

    payload = result.to_dict()
    assert payload["holdout_report_results"]["clf/accuracy"] == pytest.approx(1.0)
    assert payload["report_evaluator"] is not None


@pytest.mark.parametrize(
    "splitter_cls",
    [
        GroupKFold,
        StratifiedGroupKFold,
    ],
)
def test_optuna_search_cv_group_splitters_have_no_group_leakage(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
    splitter_cls,
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

    cv = make_optuna_search_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        splitter_cls=splitter_cls,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=None,
    )

    result = cv.run(tiny_dataset, index=y, groups=groups)
    trial = result.selected_trial_result()

    for fold in trial.fold_results:
        train_groups = {groups[i] for i in fold.train_indices}
        val_groups = {groups[i] for i in fold.val_indices}
        assert train_groups.isdisjoint(val_groups)

    assert sorted(trial.aggregate_oof_sample_indices) == sorted(result.search_pool_indices)


def test_optuna_search_cv_routes_parameters_into_real_model_and_trainer_specs(
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

    cv = make_optuna_search_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([4], "categorical"),
        }),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=None,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    assert result.final_model_spec is not None
    assert result.final_trainer_spec is not None
    assert result.final_model_spec.backbone.kwargs["scale_factor"] == 1.0
    assert result.final_trainer_spec.config.max_epochs == 4
    assert result.best_params["model/backbone/kwargs/scale_factor"] == 1.0
    assert result.best_params["trainer/config/max_epochs"] == 4


def test_optuna_search_cv_handles_minimize_selection_metric_correctly(
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

    cv = make_optuna_search_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=None,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    assert result.selection_metric_name == "dataset:error_rate"
    assert result.selection_metric_direction == "maximize"
    assert result.best_metric == pytest.approx(0.0)
    assert result.best_selection_score == pytest.approx(0.0)


def test_optuna_search_cv_rebuilds_final_model_and_trainer_and_preserves_calibrator_fit(
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

    cv = make_optuna_search_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=None,
        keep_final_model_state_dict_cpu=False,  # force path-based reconstruction
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    rebuilt_model = result.rebuild_final_model(device="cpu")
    phead = rebuilt_model.prediction_heads["clf"]
    calibrator = phead.calibrator
    assert calibrator is not None
    assert int(calibrator.fit_calls.item()) == 1
    assert int(calibrator.last_num_samples.item()) == len(result.search_pool_indices)
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

    rebuilt_trainer = result.rebuild_final_trainer(device="cpu")
    assert rebuilt_trainer is not None


def test_optuna_search_cv_result_is_pickleable_and_reconstruction_survives_roundtrip(
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

    cv = make_optuna_search_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=None,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    blob = pickle.dumps(result)
    restored: OptunaSearchCVResult = pickle.loads(blob)

    model = restored.rebuild_final_model(device="cpu")
    trainer = restored.rebuild_final_trainer(device="cpu")
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


def test_optuna_search_cv_final_model_reconstruction_is_prediction_identical(
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

    cv = make_optuna_search_cv(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=None,
        keep_final_model_state_dict_cpu=True,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    assert result.final_model_spec is not None
    assert result.final_model_state_dict_cpu is not None
    assert result.final_model_state_dict_path is not None

    rebuilt_default = result.rebuild_final_model(device="cpu")
    rebuilt_from_cpu = TorchkitModelFactory.build(
        result.final_model_spec,
        state_dict=result.final_model_state_dict_cpu,
        device="cpu",
    )
    rebuilt_from_path = TorchkitModelFactory.build(
        result.final_model_spec,
        state_dict_path=result.final_model_state_dict_path,
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


class FaultInjectingOptunaSearchCV(OptunaSearchCV):
    """
    Forces one pruned and one failed trial before a real success.
    """
    def _run_single_trial(
        self,
        *,
        trial,
        search_dataset,
        search_index,
        search_groups,
        search_original_indices,
    ):
        _ = self.suggest_parameters(trial, self.parameter_grid)

        if trial.number == 0:
            raise optuna.TrialPruned("synthetic prune for bookkeeping test")

        if trial.number == 1:
            raise RuntimeError("synthetic failure for bookkeeping test")

        return super()._run_single_trial(
            trial=trial,
            search_dataset=search_dataset,
            search_index=search_index,
            search_groups=search_groups,
            search_original_indices=search_original_indices,
        )


def test_optuna_search_cv_logs_failed_and_pruned_trials_with_params_and_tracebacks(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    import optuna

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

    cv = FaultInjectingOptunaSearchCV(
        model_spec=model_spec,
        trainer_spec=trainer_spec,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        splitter_cls=StratifiedKFold,
        dataloader_factory=lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=2, shuffle=shuffle),
        n_trials=1,
        max_trial_attempts=4,
        n_splits=2,
        shuffle=False,
        random_state=None,
        calibrate=True,
        final_model_dir=str(tmp_path),
        keep_final_model_state_dict_cpu=True,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    assert result.pruned_trials == 1
    assert result.failed_trials == 1
    assert result.successful_trials == 1
    assert result.attempted_trials == 3
    assert len(result.trial_results) == 3

    pruned = [tr for tr in result.trial_results if tr.status == "PRUNED"]
    failed = [tr for tr in result.trial_results if tr.status == "FAILED"]
    success = [tr for tr in result.trial_results if tr.status == "SUCCESS"]

    assert len(pruned) == 1
    assert len(failed) == 1
    assert len(success) == 1

    assert pruned[0].params == {"model/backbone/kwargs/scale_factor": 1.0}
    assert failed[0].params == {"model/backbone/kwargs/scale_factor": 1.0}
    assert pruned[0].error_message is not None
    assert failed[0].error_message is not None
    assert pruned[0].error_traceback is not None
    assert failed[0].error_traceback is not None


def test_optuna_search_cv_is_deterministic_for_same_seed(
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

    cv1 = make_optuna_search_cv(
        model_spec=model_spec_1,
        trainer_spec=trainer_spec_1,
        splitter_cls=StratifiedGroupKFold,
        parameter_grid=ParameterGrid.from_simple({
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([2], "categorical"),
        }),
        tmp_path=tmp_path / "run1",
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=42,
    )

    cv2 = make_optuna_search_cv(
        model_spec=model_spec_2,
        trainer_spec=trainer_spec_2,
        splitter_cls=StratifiedGroupKFold,
        parameter_grid=ParameterGrid.from_simple({
            "model/backbone/kwargs/scale_factor": ([1.0], "categorical"),
            "trainer/config/max_epochs": ([2], "categorical"),
        }),
        tmp_path=tmp_path / "run2",
        n_trials=1,
        max_trial_attempts=3,
        n_splits=2,
        random_state=42,
    )

    result1 = cv1.run(tiny_dataset, index=y, groups=groups)
    result2 = cv2.run(tiny_dataset, index=y, groups=groups)

    assert result1.best_params == result2.best_params
    assert result1.best_metric == pytest.approx(result2.best_metric)
    assert result1.best_selection_score == pytest.approx(result2.best_selection_score)
    assert result1.selected_metric_mean == pytest.approx(result2.selected_metric_mean)
    assert result1.selected_metric_std == pytest.approx(result2.selected_metric_std)

    tr1 = result1.selected_trial_result()
    tr2 = result2.selected_trial_result()
    assert tr1.aggregate_oof_sample_indices == tr2.aggregate_oof_sample_indices
    for task in tr1.aggregate_oof_logits:
        assert torch.equal(tr1.aggregate_oof_logits[task], tr2.aggregate_oof_logits[task])
        assert torch.equal(tr1.aggregate_oof_targets[task], tr2.aggregate_oof_targets[task])

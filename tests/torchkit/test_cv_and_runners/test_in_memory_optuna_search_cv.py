from __future__ import annotations

import optuna
import pytest

from torchkit.data.split import StratifiedKFold
from torchkit.evaluate.select import AccuracySelectorEvaluator
from torchkit.train.cv._optuna_search_mixin import ParameterGrid
from torchkit.train.trainer import Trainer

from tests.torchkit.test_cv_and_runners.conftest import (
    make_in_memory_optuna_search_cv,
    make_model_spec,
    make_trainer_spec,
)


class _PruneFirstTrialAtStepOne(optuna.pruners.BasePruner):
    def prune(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> bool:
        del study
        return trial.number == 0 and trial.last_step == 1


def test_in_memory_optuna_search_cv_uses_fit_iter_for_folds_and_records_epoch_reports(
    tiny_dataset,
    tiny_labels_groups,
    tiny_report_evaluator,
    tmp_path,
    monkeypatch,
):
    y, _groups = tiny_labels_groups

    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=2,
    )
    trainer_spec.config.validate_every = 1

    cv = make_in_memory_optuna_search_cv(
        model_spec=make_model_spec(scale_factor=1.0),
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=1,
        n_splits=2,
        report_evaluator=tiny_report_evaluator,
        logging=True,
    )

    fit_call_count = 0
    original_fit = Trainer.fit

    def counting_fit(self, *args, **kwargs):
        nonlocal fit_call_count
        fit_call_count += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(Trainer, "fit", counting_fit)

    result = cv.run(tiny_dataset, index=y, groups=None, holdout_dataset=tiny_dataset)

    assert fit_call_count == 1

    trial = result.selected_trial_result()
    assert len(trial.intermediate_reports) == 2
    assert [report["epoch"] for report in trial.intermediate_reports] == [1, 2]
    assert all(report["aggregate_selection_score"] == pytest.approx(1.0) for report in trial.intermediate_reports)
    assert all(report["n_reporting_folds"] == 2 for report in trial.intermediate_reports)
    assert trial.pruned_epoch is None
    assert result.best_metric == pytest.approx(1.0)
    assert result.best_selection_score == pytest.approx(1.0)


def test_in_memory_optuna_search_cv_keeps_pruned_trial_reports_and_partial_fold_results(
    tiny_dataset,
    tiny_labels_groups,
    tmp_path,
):
    y, _groups = tiny_labels_groups

    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=2,
    )
    trainer_spec.config.validate_every = 1

    cv = make_in_memory_optuna_search_cv(
        model_spec=make_model_spec(scale_factor=1.0),
        trainer_spec=trainer_spec,
        splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=2,
        n_splits=2,
    )
    cv._create_study = lambda: optuna.create_study(direction="maximize", pruner=_PruneFirstTrialAtStepOne())

    result = cv.run(tiny_dataset, index=y, groups=None)

    assert result.attempted_trials == 2
    assert result.successful_trials == 1
    assert result.pruned_trials == 1
    assert len(result.trial_results) == 2

    pruned_trial = result.trial_results[0]
    assert pruned_trial.status == "PRUNED"
    assert pruned_trial.pruned_epoch == 1
    assert len(pruned_trial.intermediate_reports) == 1
    assert pruned_trial.intermediate_reports[0]["epoch"] == 1
    assert pruned_trial.intermediate_reports[0]["aggregate_selection_score"] == pytest.approx(1.0)
    assert len(pruned_trial.fold_results) == 2
    assert all(fold.best_epoch == 1 for fold in pruned_trial.fold_results)
    assert pruned_trial.aggregate_selection_score == pytest.approx(1.0)

    successful_trial = result.selected_trial_result()
    assert successful_trial.status == "SUCCESS"
    assert [report["epoch"] for report in successful_trial.intermediate_reports] == [1, 2]

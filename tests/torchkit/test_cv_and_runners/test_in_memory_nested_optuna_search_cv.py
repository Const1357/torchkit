from __future__ import annotations

import pytest

from torchkit.data.split import StratifiedKFold
from torchkit.evaluate.select import AccuracySelectorEvaluator
from torchkit.train.cv._optuna_search_mixin import ParameterGrid

from tests.torchkit.test_cv_and_runners.conftest import (
    make_in_memory_nested_cv,
    make_model_spec,
    make_trainer_spec,
)


def test_in_memory_nested_cv_uses_in_memory_inner_search_and_preserves_epoch_reports(
    tiny_dataset,
    tiny_labels_groups,
    tiny_report_evaluator,
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

    cv = make_in_memory_nested_cv(
        model_spec=make_model_spec(scale_factor=1.0),
        trainer_spec=trainer_spec,
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        parameter_grid=ParameterGrid.from_simple({"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}),
        tmp_path=tmp_path,
        n_trials=1,
        max_trial_attempts=1,
        k_outer=2,
        k_inner=2,
        report_evaluator=tiny_report_evaluator,
    )

    result = cv.run(tiny_dataset, index=y, groups=None)

    assert len(result.outer_results) == 2
    for outer in result.outer_results:
        trial = outer.inner_search_result.selected_trial_result()
        assert [report["epoch"] for report in trial.intermediate_reports] == [1, 2]
        assert all(report["aggregate_selection_score"] == pytest.approx(1.0) for report in trial.intermediate_reports)
        assert trial.pruned_epoch is None
        assert outer.best_metric == pytest.approx(1.0)
        assert outer.best_selection_score == pytest.approx(1.0)
        assert outer.outer_test_report_results is not None
        assert outer.outer_test_report_results["clf/accuracy"] == pytest.approx(1.0)

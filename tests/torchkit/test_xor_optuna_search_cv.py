from __future__ import annotations

import pytest

from torchkit.data.split import StratifiedKFold
from torchkit.train.cv._optuna_search_mixin import ParameterGrid
from torchkit.train.cv.optuna_search_cv import OptunaSearchCV

from tests.torchkit._xor_test_utils import (
    XORDataset,
    make_xor_loader,
    make_xor_model,
    make_xor_trainer_spec,
    xor_accuracy,
)


def test_xor_optuna_search_cv_learns_with_live_model_and_dataset_evaluator(tmp_path):
    dataset = XORDataset(repeats=24)
    labels = [int(dataset[i]["y"].item()) for i in range(len(dataset))]

    cv = OptunaSearchCV(
        model_spec=make_xor_model(hidden_dim=16),
        trainer_spec=make_xor_trainer_spec(lr=1e-3, max_epochs=150),
        parameter_grid=ParameterGrid.from_simple({
            "trainer/config/optimizer_kwargs/lr": ([1e-3, 1e-2, 5e-2], "categorical"),
        }),
        splitter_cls=StratifiedKFold,
        dataloader_factory=lambda ds, shuffle: make_xor_loader(ds, shuffle=shuffle, batch_size=16),
        n_trials=2,
        max_trial_attempts=4,
        n_splits=2,
        shuffle=True,
        random_state=0,
        calibrate=False,
        final_model_dir=str(tmp_path),
        keep_final_model_state_dict_cpu=True,
    )

    result = cv.run(dataset, index=labels, groups=None)

    assert result.best_metric >= 0.95
    assert result.selected_metric_mean is not None
    assert result.selected_metric_mean >= 0.95
    assert result.best_params["trainer/config/optimizer_kwargs/lr"] in {1e-3, 1e-2, 5e-2}

    rebuilt_model = result.rebuild_final_model(device="cpu")
    assert xor_accuracy(rebuilt_model, dataset) >= 0.95

from __future__ import annotations

from torchkit.data.split import StratifiedKFold
from torchkit.train.cv.nested_optuna_search_cv import NestedOptunaSearchCV

from tests.torchkit._xor_test_utils import (
    XORDataset,
    make_xor_loader,
    make_xor_model,
    make_xor_trainer_spec,
    xor_accuracy,
)


def test_xor_nested_optuna_search_cv_learns_with_live_model_and_dataset_evaluator(tmp_path):
    dataset = XORDataset(repeats=24)
    labels = [int(dataset[i]["y"].item()) for i in range(len(dataset))]

    cv = NestedOptunaSearchCV(
        model_spec=make_xor_model(hidden_dim=16),
        trainer_spec=make_xor_trainer_spec(lr=1e-3, max_epochs=150),
        parameter_grid={
            "trainer/config/optimizer_kwargs/lr": ([1e-3, 1e-2, 5e-2], "categorical"),
        },
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        dataloader_factory=lambda ds, shuffle: make_xor_loader(ds, shuffle=shuffle, batch_size=16),
        n_trials=3,
        max_trial_attempts=6,
        k_outer=2,
        k_inner=2,
        shuffle_outer=True,
        shuffle_inner=True,
        random_state=0,
        calibrate=False,
        final_model_dir=str(tmp_path),
        keep_final_model_state_dict_cpu=True,
    )

    result = cv.run(dataset, index=labels, groups=None)

    assert len(result.outer_results) == 2

    for fold_idx, outer in enumerate(result.outer_results):
        assert outer.best_metric >= 0.9
        assert outer.outer_test_metrics is not None
        assert outer.outer_test_metrics["val/xor_accuracy"] >= 0.9

        rebuilt_model = result.rebuild_final_model(fold_idx, device="cpu")
        assert xor_accuracy(
            rebuilt_model,
            dataset,
            indices=outer.outer_test_indices,
        ) >= 0.9

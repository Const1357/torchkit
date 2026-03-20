from __future__ import annotations

import pytest

from torchkit.train.factory import TrainerFactory

from tests.torchkit._xor_test_utils import (
    XORDataset,
    make_xor_loader,
    make_xor_model,
    make_xor_trainer_spec,
    rebuild_model_accuracy_from_spec,
    xor_accuracy,
)


def test_xor_plain_training_learns_with_dataset_evaluator():
    dataset = XORDataset(repeats=32)
    model = make_xor_model(hidden_dim=16)
    trainer_spec = make_xor_trainer_spec(lr=1e-2, max_epochs=200)

    trainer = TrainerFactory.build(trainer_spec, model=model)
    train_loader = make_xor_loader(dataset, shuffle=True, batch_size=16)
    val_loader = make_xor_loader(dataset, shuffle=False, batch_size=16)

    trainer.fit(train_loader, val_loader)

    assert trainer.state.best_metric is not None
    assert trainer.state.best_metric >= 0.95
    assert any(log.get("val/xor_accuracy", 0.0) >= 0.95 for log in trainer.state.val_logs)
    assert xor_accuracy(trainer.model, dataset) >= 0.95
    assert rebuild_model_accuracy_from_spec(trainer.model, dataset) >= 0.95


def test_xor_plain_training_can_overfit_tiny_dataset():
    dataset = XORDataset(repeats=1)
    model = make_xor_model(hidden_dim=16)
    trainer_spec = make_xor_trainer_spec(lr=5e-2, max_epochs=400)

    trainer = TrainerFactory.build(trainer_spec, model=model)
    train_loader = make_xor_loader(dataset, shuffle=True, batch_size=4)
    val_loader = make_xor_loader(dataset, shuffle=False, batch_size=4)

    trainer.fit(train_loader, val_loader)

    assert trainer.state.best_metric is not None
    assert trainer.state.best_metric == pytest.approx(1.0)
    assert xor_accuracy(trainer.model, dataset) == pytest.approx(1.0)

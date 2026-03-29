from __future__ import annotations

import os
from pathlib import Path
import tempfile

import optuna
import pytest
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchkit.distributed import DDPStrategy, DistributedConfig, DistributedContext
from torchkit.evaluate.select.bundle import BundleSelectorEvaluator
from torchkit.objectives.relational import CELoss
from torchkit.train.trainer import Trainer, TrainerConfig

from .test_trainer import (
    BatchAccuracyEvaluator,
    DatasetAccuracyEvaluator,
    DictClassificationDataset,
    DummyTrial,
    make_classification_model,
)


def _ddp_test_loader(dataset: DictClassificationDataset, *, rank: int, world_size: int, batch_size: int) -> DataLoader:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)


def _build_ddp_trainer(*, init_method: str) -> tuple[Trainer, DDPStrategy]:
    torch.manual_seed(7)
    strategy = DDPStrategy(
        config=DistributedConfig(
            enabled=True,
            backend="gloo",
            init_method=init_method,
        ),
        context=DistributedContext.from_env(),
    )
    trainer = Trainer(
        model=make_classification_model(with_prediction_head=True),
        objective=CELoss(
            input_path="clf/logits",
            target_path="batch/y",
            reduction="mean",
        ),
        selector_evaluator=BundleSelectorEvaluator(
            dataset_evaluator=DatasetAccuracyEvaluator(),
            batch_evaluator=BatchAccuracyEvaluator(),
        ),
        config=TrainerConfig(
            device="cpu",
            max_epochs=3,
            validate_every=2,
            optimizer_cls=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.1},
            random_seed=7,
        ),
        distributed_strategy=strategy,
    )
    return trainer, strategy


def _configure_rank_env(*, rank: int, world_size: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")


def _ddp_train_worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    _configure_rank_env(rank=rank, world_size=world_size)
    trainer, strategy = _build_ddp_trainer(init_method=init_method)

    train_loader = _ddp_test_loader(
        DictClassificationDataset(12),
        rank=rank,
        world_size=world_size,
        batch_size=2,
    )
    val_loader = _ddp_test_loader(
        DictClassificationDataset(8),
        rank=rank,
        world_size=world_size,
        batch_size=2,
    )

    after_validation_epochs: list[int] = []
    try:
        trainer.fit(
            train_loader,
            val_loader,
            validate_every=2,
            after_validation=lambda _trainer, event: after_validation_epochs.append(event.epoch),
        )
        payload = {
            "rank": rank,
            "train_logs": trainer.state.train_logs,
            "val_logs": trainer.state.val_logs,
            "best_epoch": trainer.state.best_epoch,
            "best_metric": trainer.state.best_metric,
            "after_validation_epochs": after_validation_epochs,
            "oof_logits_shape": tuple(trainer.state.oof_logits["clf"].shape),
            "oof_targets_shape": tuple(trainer.state.oof_targets["clf"].shape),
            "oof_logits_sum": float(trainer.state.oof_logits["clf"].sum().item()),
            "oof_targets_sum": float(trainer.state.oof_targets["clf"].sum().item()),
        }
        torch.save(payload, Path(output_dir) / f"rank_{rank}.pt")
    finally:
        strategy.finalize()


def _ddp_prune_worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    _configure_rank_env(rank=rank, world_size=world_size)
    trainer, strategy = _build_ddp_trainer(init_method=init_method)

    train_loader = _ddp_test_loader(
        DictClassificationDataset(12),
        rank=rank,
        world_size=world_size,
        batch_size=2,
    )
    val_loader = _ddp_test_loader(
        DictClassificationDataset(8),
        rank=rank,
        world_size=world_size,
        batch_size=2,
    )

    trial = DummyTrial(prune_on_report=True) if rank == 0 else None
    payload: dict[str, object] | None = None
    try:
        trainer.fit(train_loader, val_loader, trial=trial, validate_every=1)
    except optuna.TrialPruned:
        payload = {
            "rank": rank,
            "status": "pruned",
            "epochs_ran": trainer.state.epoch,
            "reports": [] if trial is None else list(trial.reports),
        }
    else:
        payload = {
            "rank": rank,
            "status": "completed",
            "epochs_ran": trainer.state.epoch,
            "reports": [] if trial is None else list(trial.reports),
        }
    finally:
        if payload is not None:
            torch.save(payload, Path(output_dir) / f"rank_{rank}.pt")
        strategy.finalize()


def _spawn_and_collect(
    worker,
    *,
    world_size: int = 2,
) -> list[dict[str, object]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = f"file://{Path(tmpdir) / 'ddp_init'}"
        mp.spawn(
            worker,
            args=(world_size, init_method, tmpdir),
            nprocs=world_size,
            join=True,
        )
        return [
            torch.load(Path(tmpdir) / f"rank_{rank}.pt", map_location="cpu", weights_only=False)
            for rank in range(world_size)
        ]


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_distributed_trainer_gathers_metrics_and_oof_across_ranks() -> None:
    results = _spawn_and_collect(_ddp_train_worker)

    assert len(results) == 2
    rank0, rank1 = results
    assert rank0["train_logs"] == rank1["train_logs"]
    assert rank0["val_logs"] == rank1["val_logs"]
    assert rank0["best_epoch"] == rank1["best_epoch"]
    assert rank0["best_metric"] == pytest.approx(rank1["best_metric"])
    assert rank0["after_validation_epochs"] == [2, 3]
    assert rank1["after_validation_epochs"] == [2, 3]
    assert rank0["oof_logits_shape"] == (8, 2)
    assert rank1["oof_logits_shape"] == (8, 2)
    assert rank0["oof_targets_shape"] == (8,)
    assert rank1["oof_targets_shape"] == (8,)
    assert rank0["oof_logits_sum"] == pytest.approx(rank1["oof_logits_sum"])
    assert rank0["oof_targets_sum"] == pytest.approx(rank1["oof_targets_sum"])


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_distributed_trainer_broadcasts_pruning_from_main_rank() -> None:
    results = _spawn_and_collect(_ddp_prune_worker)

    assert len(results) == 2
    assert all(result["status"] == "pruned" for result in results)
    assert all(int(result["epochs_ran"]) == 1 for result in results)
    assert len(results[0]["reports"]) == 1
    report_value, report_step = results[0]["reports"][0]
    assert isinstance(report_value, float)
    assert report_step == 1
    assert results[1]["reports"] == []

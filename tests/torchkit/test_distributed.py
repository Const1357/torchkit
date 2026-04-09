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

from torchkit.data.split import StratifiedKFold
from torchkit.distributed import DDPStrategy, DistributedConfig, DistributedContext
from torchkit.evaluate.select import AccuracySelectorEvaluator
from torchkit.evaluate.select.bundle import BundleSelectorEvaluator
from torchkit.objectives.relational import CELoss
from torchkit.train.cv._optuna_search_mixin import ParameterGrid
from torchkit.train.cv.in_memory_nested_optuna_search_cv import InMemoryNestedOptunaSearchCV
from torchkit.train.cv.in_memory_optuna_search_cv import InMemoryOptunaSearchCV
from torchkit.train.trainer import Trainer, TrainerConfig

from .test_trainer import (
    BatchAccuracyEvaluator,
    DatasetAccuracyEvaluator,
    DictClassificationDataset,
    DummyTrial,
    make_classification_model,
)
from .test_cv_and_runners.conftest import (
    TinyClassificationDataset,
    make_labels_and_groups,
    make_model_spec,
    make_trainer_spec,
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


class _CountingStudyInMemoryOptunaSearchCV(InMemoryOptunaSearchCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_study_calls = 0

    def _create_study(self):
        self.create_study_calls += 1
        return super()._create_study()


class _AlwaysPruningInMemoryOptunaSearchCV(_CountingStudyInMemoryOptunaSearchCV):
    def _run_single_trial_with_params(
        self,
        *,
        trial_number,
        params,
        search_dataset,
        search_index,
        search_groups,
        search_original_indices,
        trial=None,
    ):
        raise optuna.TrialPruned("synthetic distributed prune")


class _CountingStudyInMemoryNestedOptunaSearchCV(InMemoryNestedOptunaSearchCV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner_searches: list[_CountingStudyInMemoryOptunaSearchCV] = []

    @property
    def create_study_calls(self) -> int:
        return sum(inner.create_study_calls for inner in self._inner_searches)

    def _build_inner_search(self, *, outer_fold: int) -> _CountingStudyInMemoryOptunaSearchCV:
        inner = _CountingStudyInMemoryOptunaSearchCV(
            model_spec=make_model_spec(scale_factor=1.0),
            trainer_spec=self.trainer_spec,
            parameter_grid=self.parameter_grid,
            splitter_cls=self.inner_splitter_cls,
            dataloader_factory=self.dataloader_factory,
            n_trials=self.n_trials,
            max_trial_attempts=self.max_trial_attempts,
            n_splits=self.k_inner if self.k_inner is not None else 0,
            shuffle=self.shuffle_inner,
            random_state=self.random_state,
            calibrate=self.calibrate,
            report_evaluator=self.report_evaluator,
            logging=self.logging,
            _log_root_dir=None,
            final_model_dir=self._outer_fold_model_dir(outer_fold),
            keep_final_model_state_dict_cpu=self.keep_final_model_state_dict_cpu,
        )
        self._inner_searches.append(inner)
        return inner


def _distributed_search_worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    _configure_rank_env(rank=rank, world_size=world_size)
    strategy = DDPStrategy(
        config=DistributedConfig(
            enabled=True,
            backend="gloo",
            init_method=init_method,
        ),
        context=DistributedContext.from_env(),
    )

    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=1,
    )
    trainer_spec.config.device = "cpu"
    trainer_spec.config.validate_every = 1
    trainer_spec.distributed_strategy = strategy

    def dataloader_factory(ds, shuffle):
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        return DataLoader(ds, batch_size=2, sampler=sampler, shuffle=False)

    cv = _CountingStudyInMemoryOptunaSearchCV(
        model_spec=make_model_spec(scale_factor=1.0),
        trainer_spec=trainer_spec,
        parameter_grid=ParameterGrid.from_simple(
            {"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}
        ),
        splitter_cls=StratifiedKFold,
        dataloader_factory=dataloader_factory,
        n_trials=1,
        max_trial_attempts=1,
        n_splits=2,
        logging=False,
        final_model_dir=output_dir,
    )

    dataset = TinyClassificationDataset()
    labels, _groups = make_labels_and_groups()
    payload: dict[str, object]
    try:
        result = cv.run(dataset, index=labels, groups=None)
        payload = {
            "ok": True,
            "rank": rank,
            "create_study_calls": cv.create_study_calls,
            "attempted_trials": result.attempted_trials,
            "successful_trials": result.successful_trials,
            "trial_numbers": [trial_result.trial_number for trial_result in result.trial_results],
            "trial_statuses": [trial_result.status for trial_result in result.trial_results],
            "best_trial_number": result.best_trial_number,
        }
    except Exception as exc:
        import traceback

        payload = {
            "ok": False,
            "rank": rank,
            "create_study_calls": cv.create_study_calls,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        torch.save(payload, Path(output_dir) / f"search_rank_{rank}.pt")
        strategy.finalize()


def _distributed_nested_search_worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    _configure_rank_env(rank=rank, world_size=world_size)
    strategy = DDPStrategy(
        config=DistributedConfig(
            enabled=True,
            backend="gloo",
            init_method=init_method,
        ),
        context=DistributedContext.from_env(),
    )

    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=1,
    )
    trainer_spec.config.device = "cpu"
    trainer_spec.config.validate_every = 1
    trainer_spec.distributed_strategy = strategy

    def dataloader_factory(ds, shuffle):
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        return DataLoader(ds, batch_size=2, sampler=sampler, shuffle=False)

    cv = _CountingStudyInMemoryNestedOptunaSearchCV(
        model_spec=make_model_spec(scale_factor=1.0),
        trainer_spec=trainer_spec,
        parameter_grid=ParameterGrid.from_simple(
            {"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}
        ),
        outer_splitter_cls=StratifiedKFold,
        inner_splitter_cls=StratifiedKFold,
        dataloader_factory=dataloader_factory,
        n_trials=1,
        max_trial_attempts=1,
        k_outer=2,
        k_inner=2,
        logging=False,
        final_model_dir=output_dir,
    )

    dataset = TinyClassificationDataset()
    labels, _groups = make_labels_and_groups()
    payload: dict[str, object]
    try:
        result = cv.run(dataset, index=labels, groups=None)
        payload = {
            "ok": True,
            "rank": rank,
            "create_study_calls": cv.create_study_calls,
            "n_outer_results": len(result.outer_results),
            "outer_best_trial_numbers": [outer.best_trial_number for outer in result.outer_results],
            "outer_best_metrics": [outer.best_metric for outer in result.outer_results],
            "outer_best_selection_scores": [outer.best_selection_score for outer in result.outer_results],
        }
    except Exception as exc:
        import traceback

        payload = {
            "ok": False,
            "rank": rank,
            "create_study_calls": cv.create_study_calls,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        torch.save(payload, Path(output_dir) / f"nested_search_rank_{rank}.pt")
        strategy.finalize()


def _distributed_exhausted_search_worker(rank: int, world_size: int, init_method: str, output_dir: str) -> None:
    _configure_rank_env(rank=rank, world_size=world_size)
    strategy = DDPStrategy(
        config=DistributedConfig(
            enabled=True,
            backend="gloo",
            init_method=init_method,
        ),
        context=DistributedContext.from_env(),
    )

    trainer_spec = make_trainer_spec(
        evaluator=AccuracySelectorEvaluator(
            score_key="clf/logits",
            target_key="batch/y",
            name="classification",
        ),
        max_epochs=1,
    )
    trainer_spec.config.device = "cpu"
    trainer_spec.config.validate_every = 1
    trainer_spec.distributed_strategy = strategy

    def dataloader_factory(ds, shuffle):
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        return DataLoader(ds, batch_size=2, sampler=sampler, shuffle=False)

    cv = _AlwaysPruningInMemoryOptunaSearchCV(
        model_spec=make_model_spec(scale_factor=1.0),
        trainer_spec=trainer_spec,
        parameter_grid=ParameterGrid.from_simple(
            {"model/backbone/kwargs/scale_factor": ([1.0], "categorical")}
        ),
        splitter_cls=StratifiedKFold,
        dataloader_factory=dataloader_factory,
        n_trials=1,
        max_trial_attempts=1,
        n_splits=2,
        logging=False,
        final_model_dir=output_dir,
    )

    dataset = TinyClassificationDataset()
    labels, _groups = make_labels_and_groups()
    payload: dict[str, object]
    try:
        cv.run(dataset, index=labels, groups=None)
        payload = {
            "ok": True,
            "rank": rank,
            "create_study_calls": cv.create_study_calls,
        }
    except Exception as exc:
        import traceback

        payload = {
            "ok": False,
            "rank": rank,
            "create_study_calls": cv.create_study_calls,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        torch.save(payload, Path(output_dir) / f"exhausted_search_rank_{rank}.pt")
        strategy.finalize()


def _spawn_and_collect(
    worker,
    *,
    world_size: int = 2,
    file_prefix: str = "rank_",
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
            torch.load(Path(tmpdir) / f"{file_prefix}{rank}.pt", map_location="cpu", weights_only=False)
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


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_distributed_in_memory_optuna_search_creates_study_only_on_main_rank() -> None:
    results = _spawn_and_collect(_distributed_search_worker, file_prefix="search_rank_")

    assert len(results) == 2
    rank0, rank1 = results
    assert rank0["ok"], rank0.get("traceback")
    assert rank1["ok"], rank1.get("traceback")
    assert rank0["create_study_calls"] == 1
    assert rank1["create_study_calls"] == 0
    assert rank0["attempted_trials"] == 1
    assert rank1["attempted_trials"] == 1
    assert rank0["successful_trials"] == 1
    assert rank1["successful_trials"] == 1
    assert rank0["trial_numbers"] == [0]
    assert rank1["trial_numbers"] == [0]
    assert rank0["trial_statuses"] == ["SUCCESS"]
    assert rank1["trial_statuses"] == ["SUCCESS"]
    assert rank0["best_trial_number"] == 0
    assert rank1["best_trial_number"] == 0


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_distributed_in_memory_nested_optuna_search_uses_main_rank_studies_and_agrees_across_ranks() -> None:
    results = _spawn_and_collect(_distributed_nested_search_worker, file_prefix="nested_search_rank_")

    assert len(results) == 2
    rank0, rank1 = results
    assert rank0["ok"], rank0.get("traceback")
    assert rank1["ok"], rank1.get("traceback")
    assert rank0["create_study_calls"] == 2
    assert rank1["create_study_calls"] == 0
    assert rank0["n_outer_results"] == 2
    assert rank1["n_outer_results"] == 2
    assert rank0["outer_best_trial_numbers"] == [0, 0]
    assert rank1["outer_best_trial_numbers"] == [0, 0]
    assert rank0["outer_best_metrics"] == pytest.approx([1.0, 1.0])
    assert rank1["outer_best_metrics"] == pytest.approx([1.0, 1.0])
    assert rank0["outer_best_selection_scores"] == pytest.approx([1.0, 1.0])
    assert rank1["outer_best_selection_scores"] == pytest.approx([1.0, 1.0])


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_distributed_in_memory_optuna_search_broadcasts_attempt_exhaustion_to_all_ranks() -> None:
    results = _spawn_and_collect(_distributed_exhausted_search_worker, file_prefix="exhausted_search_rank_")

    assert len(results) == 2
    rank0, rank1 = results
    assert not rank0["ok"], rank0.get("traceback")
    assert not rank1["ok"], rank1.get("traceback")
    assert rank0["create_study_calls"] == 1
    assert rank1["create_study_calls"] == 0
    assert "OptunaSearchCV produced no successful trials." in rank0["error"]
    assert "OptunaSearchCV produced no successful trials." in rank1["error"]
    assert "synthetic distributed prune" in rank0["error"]
    assert "synthetic distributed prune" in rank1["error"]

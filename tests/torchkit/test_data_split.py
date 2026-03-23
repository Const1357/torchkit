from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset, Subset

from torchkit.data._dataset import DatasetSplit, DatasetSubsetView, TorchkitDataset
from torchkit.data.split import (
    train_test_split,
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)


class DummyDataset(Dataset):
    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int):
        return {"x": torch.tensor([idx], dtype=torch.float32), "y": idx}


@pytest.fixture
def dataset() -> DummyDataset:
    return DummyDataset(12)


class DummyTorchkitDataset(TorchkitDataset):
    def __init__(self, n: int):
        self._n = n

    def __len__(self) -> int:
        return self._n

    def my_getitem(self, idx: int):
        return {"x": torch.tensor([idx], dtype=torch.float32), "y": idx}


class RecordingTorchkitDataset(DummyTorchkitDataset):
    def __init__(self, n: int):
        super().__init__(n)
        self.subset_calls: list[tuple[list[int], DatasetSplit]] = []

    def subset(self, indices, *, split=DatasetSplit.FULL):
        split = DatasetSplit(split)
        self.subset_calls.append((list(indices), split))
        return super().subset(indices, split=split)


@pytest.fixture
def torchkit_dataset() -> DummyTorchkitDataset:
    return DummyTorchkitDataset(12)


def _subset_indices(subset: Subset) -> list[int]:
    return list(subset.indices)


def _dataset_indices(dataset) -> list[int]:
    if isinstance(dataset, DatasetSubsetView):
        return list(dataset.indices)
    if isinstance(dataset, Subset):
        return list(dataset.indices)
    raise TypeError(f"Unsupported dataset slice type: {type(dataset).__name__}")


def test_train_test_split_returns_subsets(dataset: DummyDataset):
    train_ds, test_ds = train_test_split(
        dataset,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    )

    assert isinstance(train_ds, Subset)
    assert isinstance(test_ds, Subset)

    train_idx = set(_subset_indices(train_ds))
    test_idx = set(_subset_indices(test_ds))

    assert len(train_idx) == 9
    assert len(test_idx) == 3
    assert train_idx.isdisjoint(test_idx)
    assert train_idx | test_idx == set(range(len(dataset)))


def test_train_test_split_uses_dataset_hook_for_torchkit_dataset(torchkit_dataset: DummyTorchkitDataset):
    train_ds, test_ds = train_test_split(
        torchkit_dataset,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    )

    assert isinstance(train_ds, DatasetSubsetView)
    assert isinstance(test_ds, DatasetSubsetView)
    assert train_ds.split is DatasetSplit.TRAIN
    assert test_ds.split is DatasetSplit.TEST

    train_idx = set(_dataset_indices(train_ds))
    test_idx = set(_dataset_indices(test_ds))
    assert train_idx.isdisjoint(test_idx)
    assert train_idx | test_idx == set(range(len(torchkit_dataset)))


def test_train_test_split_is_reproducible(dataset: DummyDataset):
    train_a, test_a = train_test_split(
        dataset,
        test_size=0.25,
        shuffle=True,
        random_state=123,
    )
    train_b, test_b = train_test_split(
        dataset,
        test_size=0.25,
        shuffle=True,
        random_state=123,
    )

    assert _subset_indices(train_a) == _subset_indices(train_b)
    assert _subset_indices(test_a) == _subset_indices(test_b)


def test_train_test_split_with_stratify_preserves_class_balance_reasonably(dataset: DummyDataset):
    y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    train_ds, test_ds = train_test_split(
        dataset,
        test_size=0.25,
        shuffle=True,
        random_state=42,
        stratify=y,
    )

    train_labels = [y[i] for i in _subset_indices(train_ds)]
    test_labels = [y[i] for i in _subset_indices(test_ds)]

    assert train_labels.count(0) == 4
    assert train_labels.count(1) == 5
    assert test_labels.count(0) == 2
    assert test_labels.count(1) == 1


def test_stratified_kfold_preserves_partition_and_sizes(torchkit_dataset: DummyTorchkitDataset):
    y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(splitter.split(torchkit_dataset, y=y))

    assert len(folds) == 3

    seen_val = []
    for train_ds, val_ds in folds:
        assert isinstance(train_ds, DatasetSubsetView)
        assert isinstance(val_ds, DatasetSubsetView)
        assert train_ds.split is DatasetSplit.TRAIN
        assert val_ds.split is DatasetSplit.VAL

        train_idx = set(_dataset_indices(train_ds))
        val_idx = set(_dataset_indices(val_ds))

        assert train_idx.isdisjoint(val_idx)
        assert train_idx | val_idx == set(range(len(torchkit_dataset)))
        assert len(val_idx) == 4

        seen_val.extend(_dataset_indices(val_ds))

    assert sorted(seen_val) == list(range(len(torchkit_dataset)))


def test_stratified_kfold_val_class_balance_is_reasonable(torchkit_dataset: DummyTorchkitDataset):
    y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    for _, val_ds in splitter.split(torchkit_dataset, y=y):
        val_labels = [y[i] for i in _dataset_indices(val_ds)]
        assert val_labels.count(0) == 2
        assert val_labels.count(1) == 2


def test_group_kfold_never_splits_groups_across_train_and_val(torchkit_dataset: DummyTorchkitDataset):
    y = [0] * len(torchkit_dataset)
    groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=42)

    for train_ds, val_ds in splitter.split(torchkit_dataset, y=y, groups=groups):
        train_idx = _dataset_indices(train_ds)
        val_idx = _dataset_indices(val_ds)

        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}

        assert train_groups.isdisjoint(val_groups)
        assert set(train_idx).isdisjoint(set(val_idx))
        assert set(train_idx) | set(val_idx) == set(range(len(torchkit_dataset)))


def test_stratified_group_kfold_preserves_partition_and_group_integrity(torchkit_dataset: DummyTorchkitDataset):
    y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
    groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(splitter.split(torchkit_dataset, y=y, groups=groups))

    assert len(folds) == 3

    seen_val = []
    for train_ds, val_ds in folds:
        train_idx = _dataset_indices(train_ds)
        val_idx = _dataset_indices(val_ds)

        assert set(train_idx).isdisjoint(set(val_idx))
        assert set(train_idx) | set(val_idx) == set(range(len(torchkit_dataset)))

        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        assert train_groups.isdisjoint(val_groups)

        seen_val.extend(val_idx)

    assert sorted(seen_val) == list(range(len(torchkit_dataset)))


def test_splitters_pass_expected_split_hooks_to_torchkit_dataset():
    dataset = RecordingTorchkitDataset(12)
    y = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    groups = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    list(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(dataset, y=y))
    list(GroupKFold(n_splits=3, shuffle=True, random_state=0).split(dataset, y=y, groups=groups))
    list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0).split(dataset, y=y, groups=groups))

    assert len(dataset.subset_calls) == 18
    assert all(split in {DatasetSplit.TRAIN, DatasetSplit.VAL} for _, split in dataset.subset_calls)


def test_group_kfold_requires_groups_argument(dataset: DummyDataset):
    splitter = GroupKFold(n_splits=3)
    y = [0] * len(dataset)

    with pytest.raises(TypeError):
        list(splitter.split(dataset, y=y))


def test_stratified_group_kfold_requires_groups_argument(dataset: DummyDataset):
    splitter = StratifiedGroupKFold(n_splits=3)
    y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]

    with pytest.raises(TypeError):
        list(splitter.split(dataset, y=y))

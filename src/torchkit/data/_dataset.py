from __future__ import annotations

from typing import Any, Mapping, Sequence, final
from abc import ABC, abstractmethod
from enum import Enum

from torch.utils.data import Dataset
from torch import Tensor


class DatasetSplit(str, Enum):
    FULL = "full"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    HOLDOUT = "holdout"


class TorchkitDataset(Dataset, ABC):
    """
    Base dataset class with validation and split-aware dataset hooks.

    Subclasses should:

    1. implement `__init__`
    2. implement `__len__`
    3. implement `my_getitem(self, index)`

    `my_getitem` must return a mapping with at least:

        {"x": Tensor}

    Other keys (e.g. targets, metadata) are optional.

    Split-aware usage:

    - `subset(indices, split=...)` is the framework hook for creating
      split-specific dataset views.
    - `split` is a semantic label describing how the subset will be used,
      not just which samples it contains.
    - Typical values are `DatasetSplit.TRAIN`, `DatasetSplit.VAL`,
      `DatasetSplit.TEST`, and `DatasetSplit.HOLDOUT`.
    - Datasets can override `subset(...)` to return application-specific
      child datasets with split-dependent behavior, such as:
      - online augmentation only for training
      - deterministic preprocessing for validation / test
      - split-specific sampling or caching policy

    By default, `subset(...)` returns a generic `DatasetSubsetView`.
    Datasets that override `subset(...)` should also override
    `resolve_original_indices()` so CV / logging code can recover sample
    indices in the original root dataset.
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def my_getitem(self, index: int) -> Mapping[str, Any]:
        raise NotImplementedError

    def subset(
        self,
        indices: Sequence[int],
        *,
        split: DatasetSplit | str = DatasetSplit.FULL,
    ) -> "TorchkitDataset":
        return DatasetSubsetView(
            dataset=self,
            indices=list(indices),
            split=DatasetSplit(split),
        )

    def resolve_original_indices(self) -> list[int]:
        return list(range(len(self)))

    @final
    def __getitem__(self, index: int) -> Mapping[str, Any]:
        item = self.my_getitem(index)

        if not isinstance(item, Mapping):
            raise TypeError(
                f"Dataset item at index {index} must be a mapping, "
                f"got {type(item).__name__}."
            )

        if "x" not in item:
            raise KeyError(
                f"Dataset item at index {index} must contain key 'x'. "
                f"Got keys: {list(item.keys())}"
            )

        x = item["x"]

        if not isinstance(x, Tensor):
            raise TypeError(
                f"Dataset item['x'] must be a torch.Tensor, "
                f"got {type(x).__name__}."
            )

        return item


class DatasetSubsetView(TorchkitDataset):
    def __init__(
        self,
        *,
        dataset: Dataset,
        indices: Sequence[int],
        split: DatasetSplit = DatasetSplit.FULL,
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.split = DatasetSplit(split)

    def __len__(self) -> int:
        return len(self.indices)

    def my_getitem(self, index: int) -> Mapping[str, Any]:
        return self.dataset[self.indices[index]]

    def resolve_original_indices(self) -> list[int]:
        if isinstance(self.dataset, TorchkitDataset):
            parent_indices = self.dataset.resolve_original_indices()
        else:
            parent_indices = list(range(len(self.dataset)))
        return [parent_indices[i] for i in self.indices]
    
# NOTE: Dataset __getitem__ should:
# 1. always return stable keys
# 2. missing masks should be encoded as masks with None values
# 3. [batch]["x"] must exist and be a tensor
# 4. Targets from batch should be under batch[task][y/target/targets/label/labels]
#    or at least [batch][y/target/targets/label/labels] for single-task.
# 5. [batch]["x"] and [batch]["y/target/targets/label/labels"] should have the same batch size (dim=0)

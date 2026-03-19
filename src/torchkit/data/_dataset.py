from __future__ import annotations

from typing import Any, Mapping, final
from abc import ABC, abstractmethod

from torch.utils.data import Dataset
from torch import Tensor


class TorchkitDataset(Dataset, ABC):
    """
    Base dataset class with validation.

    Subclasses should:

    1. implement `__init__`
    2. implement `__len__`
    3. implement `my_getitem(self, index)`

    `my_getitem` must return a mapping with at least:

        {"x": Tensor}

    Other keys (e.g. targets, metadata) are optional.
    """

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def my_getitem(self, index: int) -> Mapping[str, Any]:
        raise NotImplementedError

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
    
# NOTE: Dataset __getitem__ should:
# 1. always return stable keys
# 2. missing masks should be encoded as masks with None values
# 3. [batch]["x"] must exist and be a tensor
# 4. Targets from batch should be under batch[task][y/target/targets/label/labels]
#    or at least [batch][y/target/targets/label/labels] for single-task.
# 5. [batch]["x"] and [batch]["y/target/targets/label/labels"] should have the same batch size (dim=0)
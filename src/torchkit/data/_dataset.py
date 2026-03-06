from __future__ import annotations
from typing import Any

from torch.utils.data import Dataset
from abc import abstractmethod, ABC

from  torch import Tensor

import torch

# abstract class that performs validation of getitem output (ensure dict of tensors)
class TorchkitDataset(Dataset, ABC):

    """Subclasses of TorchkitDataset should:
    1. Implement `__init__(self, ...)`.
    2. implement `__len__(self)`.
    3. implement `my_getitem(self, index)` and return a dict. (`__getitem__` performs validation checks)
    
    ### *Note*
    Suggestion: in your `__init__`, define the preprocessing and augmentation pipelines."""

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def my_getitem(self, index) -> dict[str, Any]:
        raise NotImplementedError("Subclasses of TorchkitDataset should implement `my_getitem(self, index)` method.")

    @abstractmethod
    def __getitem__(self, index):
        # implement getitem here.
        item = self.my_getitem(index)
        if not isinstance(item, dict):
            raise TypeError(f"`__getitem__` should always return dict, but got {type(item).__name__}")
        if not "x" in item.keys():
            raise KeyError(f"`my_getitem` should always return dict with key 'x', but got keys {item.keys()}")
        if not isinstance(item["x"], Tensor):
            raise TypeError(f"`my_getitem` 'x' should be of type torch.Tensor, but got type {type(item['x']).__name__}")
        
        # we cannot enforce other checks since some datasets might not contain targets.
        return item
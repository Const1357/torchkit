from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Optional, final, Iterator, Tuple
import numpy as np

from sklearn.model_selection import train_test_split as sklearn_train_test_split

from sklearn.model_selection import GroupKFold as SklearnGroupKFold
from sklearn.model_selection import StratifiedKFold as SklearnStratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold as SklearnStratifiedGroupKFold

from torchkit.data import TorchkitDataset

from torch.utils.data import Subset

# TODO: when needed, extend this file to wrap more sklearn splitters.

def train_test_split(
    dataset: TorchkitDataset,
    index: Any,
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: Optional[int] = None,
):
    """Wrapper around `sklearn.model_selection.train_test_split` that returns the split Subsets of the original dataset."""
    train_idx, test_idx = sklearn_train_test_split(
        range(len(index)),
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
    )

    return Subset(dataset, train_idx), Subset(dataset, test_idx)



class KFoldSplitter(ABC):
    """Base class for K-Fold splitters."""
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    @abstractmethod
    def split(self, dataset: TorchkitDataset, index: Any, groups: Optional[Any] = None):
        """Split the dataset into training and validation sets."""
        raise NotImplementedError("Subclasses must implement this method.")
        

class GroupKFold(KFoldSplitter):
    """`GroupKFold` is a wrapper around `sklearn.GroupKFold`."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @final
    def split(self, dataset: TorchkitDataset, index: Any, groups: Any) -> Iterator[Tuple[Subset, Subset]]:
        
        gkf = SklearnGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )  

        dummy_X = np.arange(len(index))  # dummy X, not used in splitting
        for train_idx, val_idx in gkf.split(dummy_X, index, groups):
            yield Subset(dataset, train_idx), Subset(dataset, val_idx)

class StratifiedKFold(KFoldSplitter):
    """`StratifiedKFold` is a wrapper around `sklearn.StratifiedKFold`."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @final
    def split(self, dataset: TorchkitDataset, index: Any) -> Iterator[Tuple[Subset, Subset]]:
        
        skf = SklearnStratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )  

        dummy_X = np.arange(len(index))  # dummy X, not used in splitting
        for train_idx, val_idx in skf.split(dummy_X, index):
            yield Subset(dataset, train_idx), Subset(dataset, val_idx)


class StratifiedGroupKFold(KFoldSplitter):
    """`StratifiedGroupKfold` is a wrapper around `sklearn.StratifiedGroupKFold`."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @final
    def split(self, dataset: TorchkitDataset, index: Any, groups: Any) -> Iterator[Tuple[Subset, Subset]]:
        
        sgkf = SklearnStratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )  

        dummy_X = np.arange(len(index))  # dummy X, not used in splitting
        for train_idx, val_idx in sgkf.split(dummy_X, index, groups):
            yield Subset(dataset, train_idx), Subset(dataset, val_idx)
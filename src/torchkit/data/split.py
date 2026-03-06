from __future__ import annotations

from abc import ABC, abstractmethod

from typing import Any, Optional, final
from sklearn.model_selection import GroupKFold as SklearnGroupKFold
from sklearn.model_selection import StratifiedKFold as SklearnStratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold as SklearnStratifiedGroupKFold

from torchkit.data._dataset import TorchkitDataset

from torch.utils.data import Subset

# TODO: when needed, extend this file to wrap more sklearn splitters.
# NOTE: Update the DataSplitter type hint when adding new splitters.

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
    def split(self, dataset: TorchkitDataset, index: Any, groups: Any):
        
        gkf = SklearnGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )  

        import numpy as np
        dummy_X = np.arange(len(index))  # dummy X, not used in splitting
        train_idx, val_idx =  gkf.split(dummy_X, index, groups)

        return Subset(dataset, train_idx), Subset(dataset, val_idx)

class StratifiedKFold(KFoldSplitter):
    """`StratifiedKFold` is a wrapper around `sklearn.StratifiedKFold`."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @final
    def split(self, dataset: TorchkitDataset, index: Any):
        
        skf = SklearnStratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )  

        import numpy as np
        dummy_X = np.arange(len(index))  # dummy X, not used in splitting
        train_idx, val_idx =  skf.split(dummy_X, index)

        return Subset(dataset, train_idx), Subset(dataset, val_idx)


class StratifiedGroupKFold(KFoldSplitter):
    """`StratifiedGroupKfold` is a wrapper around `sklearn.StratifiedGroupKFold`."""

    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    @final
    def split(self, dataset: TorchkitDataset, index: Any, groups: Any):
        
        sgkf = SklearnStratifiedGroupKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )  

        import numpy as np
        dummy_X = np.arange(len(index))  # dummy X, not used in splitting
        train_idx, val_idx =  sgkf.split(dummy_X, index, groups)

        return Subset(dataset, train_idx), Subset(dataset, val_idx)
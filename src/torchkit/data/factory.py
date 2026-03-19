from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from torch.utils.data import DataLoader, Dataset


@dataclass
class DataLoaderSpec:
    cls: type[DataLoader] | None = DataLoader
    kwargs: dict[str, Any] = field(default_factory=dict)
    factory: Optional[Callable[[Dataset, bool], DataLoader]] = None


class DataLoaderFactory:

    @staticmethod
    def build(
        spec: DataLoaderSpec,
        *,
        dataset: Dataset,
        shuffle: bool = False,
    ) -> DataLoader:
        if spec.factory is not None:
            loader = spec.factory(dataset, shuffle)
        else:
            if spec.cls is None:
                raise ValueError("DataLoaderSpec.cls must be specified when factory is not provided.")
            if not issubclass(spec.cls, DataLoader):
                raise TypeError(
                    f"DataLoaderSpec.cls must be a subclass of DataLoader, got {spec.cls}."
                )
            loader = spec.cls(dataset, shuffle=shuffle, **spec.kwargs)

        if not isinstance(loader, DataLoader):
            raise TypeError(
                f"DataLoader factory must return a torch.utils.data.DataLoader, got {type(loader)}."
            )

        return loader

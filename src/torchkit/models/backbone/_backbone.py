from __future__ import annotations
from typing import Any, Collection, Optional

from torch import nn, Tensor

from abc import ABC, abstractmethod


class Backbone(nn.Module, ABC):
    """Base class for backbone models.

    Contracts:
        - All backbone models must inherit from this class.
        - All backbone models must implement the `_forward_impl` method.
        - All backbone models return a dictionary of named feature maps in the implemented `_forward_impl` method. (essential for routing).
        - Must: In `_forward_impl`, support the `requested_features` argument to avoid computing unnecessary features (e.g., training-only features). You can ignore it, but it must exist in the signature.
    """

    def __init__(
        self,
        supported_features: Collection[str] = None,
    ):
        super().__init__()

        if supported_features is None:
            raise ValueError("Backbone must specify supported_features (non-None).")

        if not isinstance(supported_features, Collection):
            raise TypeError(
                f"`supported_features` must be a Collection[str], got {type(supported_features).__name__}."
            )
        for s in supported_features:
            if not isinstance(s, str):
                raise TypeError(
                    f"`supported_features` must contain only str, got {type(s).__name__}: {s!r}."
                )

        self._supported_features = set(supported_features)

    @property
    def available_features(self) -> Collection[str] | None:
        """Return the names of the features that this backbone supports."""
        return self._supported_features

    def freeze(self) -> "Backbone":
        """Freeze the backbone parameters. Does not set eval mode."""
        for param in self.parameters():
            param.requires_grad_(False)
        return self

    def unfreeze(self) -> "Backbone":
        """Unfreeze the backbone parameters. Does not set train mode"""
        for param in self.parameters():
            param.requires_grad_(True)
        return self

    def forward(
        self,
        input: dict[str, Any],
        *,
        requested_features: Optional[Collection[str]] = None,
        **kwargs,
    ) -> dict[str, Tensor]:

        if requested_features is not None:
            if not isinstance(requested_features, Collection):
                raise TypeError(
                    f"`requested_features` must be a Collection[str] or None, got {type(requested_features).__name__}."
                )
            for s in requested_features:
                if not isinstance(s, str):
                    raise TypeError(
                        f"`requested_features` must contain only str, got {type(s).__name__}: {s!r}."
                    )

            requested_features = set(requested_features)

            unsupported = requested_features - self._supported_features
            if unsupported:
                raise KeyError(
                    f"{self.__class__.__name__} does not support requested features {sorted(unsupported)}. "
                    f"Supported: {sorted(self._supported_features)}."
                )
        else:
            requested_features = self.available_features

        out: dict[str, Tensor] = self._forward_impl(input, requested_features=requested_features, **kwargs)

        if not isinstance(out, dict):
            raise TypeError(
                f"{self.__class__.__name__} forward must return dict[str, Tensor], got {type(out).__name__}."
            )
        for k, v in out.items():
            if not isinstance(k, str):
                raise TypeError(
                    f"{self.__class__.__name__} output keys must be str, got {type(k).__name__}: {k!r}."
                )
            if not isinstance(v, Tensor):
                raise TypeError(
                    f"{self.__class__.__name__} output[{k!r}] must be a Tensor, got {type(v).__name__}."
                )

        out_keys = set(out.keys())

        extra = out_keys - set(requested_features)
        if extra:
            raise KeyError(
                f"{self.__class__.__name__} returned unrequested features {sorted(extra)}. "
                f"Requested: {sorted(requested_features)}."
            )

        # Safety: prevent backbones from inventing keys
        unknown = out_keys - self._supported_features
        if unknown:
            raise KeyError(
                f"{self.__class__.__name__} returned unsupported features {sorted(unknown)}. "
                f"Supported: {sorted(self._supported_features)}."
            )

        # Core contract: all requested keys must be present in the output
        missing = set(requested_features) - out_keys
        if missing:
            raise KeyError(
                f"{self.__class__.__name__} did not return requested features {sorted(missing)}. "
                f"Returned: {sorted(out_keys)}."
            )

        return out

    @abstractmethod
    def _forward_impl(
        self,
        input: dict[str, Any],
        *,
        requested_features: Optional[Collection[str]] = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Implementation of the forward pass. Subclasses must implement this method.\\
            Assume that `requested_features` is a subset of `available_features` (validation is done in the forward method).\\
            The output *must* be a dictionary of named Tensors(feature maps). The keys of the output dictionary must include all `requested_features`."""
        raise NotImplementedError("Subclasses must implement _forward_impl")
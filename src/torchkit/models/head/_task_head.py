from __future__ import annotations
from typing import Any, Optional, Set, Tuple, List

from torch import nn, Tensor
from warnings import warn

from torchkit.models.adapters import FeatureAdapter
from torchkit.models.fuse import FuseModule

import inspect

def _supports_kwarg(mod: nn.Module, name: str) -> bool:
    try:
        sig = inspect.signature(mod.forward)
    except (TypeError, ValueError):
        return False
    return (name in sig.parameters) or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


class TaskHead(nn.Module):
    """Interface class for task heads (branches from backbone).

    Contracts:
        - All task head instances must specify the features they require from the backbone via `required_features`.

        - Fuse module is required if `required_features` has more than one element, and is ignored if
          `required_features` is a single string.
        - Fuse module fuses (e.g., concatenates) the required backbone features into a single Tensor for the adapter.

        - Feature adapter adapts the fused Tensor into the expected input format for the head module.
          Defaults to identity (may be incompatible depending on shapes).

        - Head module produces the final output for this task head.
          Defaults to identity (may be incompatible depending on shapes).
    """

    def __init__(
        self,
        *,
        required_features: str | Set[str] | Tuple[str, ...] | List[str] = None,
        fuse_module: FuseModule | nn.Module = None,
        feature_adapter: FeatureAdapter | nn.Module = None,
        head_module: nn.Module = None,
        active: bool = True,
    ):
        super().__init__()

        if required_features is None:
            raise ValueError("TaskHead must specify required_features (non-None).")

        # Normalize required features into a set[str]
        if isinstance(required_features, str):
            required_features = {required_features}
            if fuse_module is not None:
                warn(
                    f"`fuse_module` is provided to TaskHead {self.__class__.__name__} but `required_features` is a single string. "
                    f"Ignoring `fuse_module` since it is not needed for a single feature."
                )
                fuse_module = None
        elif isinstance(required_features, (set, tuple, list)):
            required_features = set(required_features)
            for s in required_features:
                if not isinstance(s, str):
                    raise TypeError(f"`required_features` must contain only str, got {type(s)}: {s!r}.")
        else:
            raise TypeError(f"`required_features` must be a str or one of (set/tuple/list) of str, got {type(required_features)}.")

        if not required_features:
            raise ValueError("`required_features` must be non-empty.")

        # Fuse rules (based on normalized set size)
        if len(required_features) > 1 and fuse_module is None:
            raise ValueError(f"`fuse_module` must be provided to TaskHead {self.__class__.__name__} when `required_features` has more than one feature.")

        if feature_adapter is None:
            warn(f"No feature_adapter provided to TaskHead {self.__class__.__name__}; using identity. "
                 f"This may cause issues if the required features do not match the head module's expected input.")
            feature_adapter = nn.Identity()

        if head_module is None:
            warn(
                f"No head_module provided to TaskHead {self.__class__.__name__}; using identity. "
                f"This may cause issues if the head module is expected to produce outputs for training or inference."
            )
            head_module = nn.Identity()

        self.fuse_module = fuse_module
        self._required_features: set[str] = required_features

        self.feature_adapter = feature_adapter
        self.head_module = head_module
        self._active = active

        self._fuse_module_supports_payload = _supports_kwarg(self.fuse_module, "payload") if self.fuse_module is not None else False
        self._feature_adapter_supports_payload = _supports_kwarg(self.feature_adapter, "payload") if self.feature_adapter is not None else False
        self._head_module_supports_payload = _supports_kwarg(self.head_module, "payload") if self.head_module is not None else False

    @property
    def required_features(self) -> set[str]:
        """Return the names of the features that this head requires."""
        return self._required_features

    @property
    def is_active(self) -> bool:
        """Return whether this head is active (i.e., should be used in forward pass)."""
        return self._active

    def enable(self) -> "TaskHead":
        """Enable this task head."""
        self._active = True
        return self

    def disable(self) -> "TaskHead":
        """Disable this head."""
        self._active = False
        return self

    def freeze(self) -> "TaskHead":
        """Freeze the head parameters. Does not set eval mode."""
        for param in self.parameters():
            param.requires_grad_(False)
        return self

    def unfreeze(self) -> "TaskHead":
        """Unfreeze the head parameters. Does not set train mode."""
        for param in self.parameters():
            param.requires_grad_(True)
        return self

    def forward(
        self, 
        features: dict[str, Tensor],
        *,
        payload: Optional[dict[str, Any]] = None,
        fuse_kwargs: Optional[dict[str, Any]] = None,
        feature_adapter_kwargs: Optional[dict[str, Any]] = None,
        head_module_kwargs: Optional[dict[str, Any]] = None
    ) -> Tensor | dict[str, Tensor] | None:
        if not self._active:
            return None  # caller should skip inactive heads
        
        fuse_kwargs = fuse_kwargs or {}
        feature_adapter_kwargs = feature_adapter_kwargs or {}
        head_module_kwargs = head_module_kwargs or {}

        keys = set(features.keys())
        missing = self._required_features - keys
        if missing:
            raise KeyError(
                f"TaskHead {self.__class__.__name__} is missing required backbone features {sorted(missing)}. "
                f"Available (this call): {sorted(keys)}."
            )

        # Select required features
        selected_features = {k: features[k] for k in self._required_features}

        if self.fuse_module is not None:
            if self._fuse_module_supports_payload and payload is not None:
                x = self.fuse_module(selected_features, payload=payload, **fuse_kwargs)
            else:
                x = self.fuse_module(selected_features, **fuse_kwargs)
        else: # no fuse module
            if len(selected_features) > 1:
                raise RuntimeError(f"TaskHead {self.__class__.__name__} has multiple required features but no `fuse_module` to combine them.")
            x = next(iter(selected_features.values()))  # get the only required feature


        if not isinstance(x, Tensor):
            if self.fuse_module is not None:
                raise TypeError(f"After fuse_module, expected a Tensor but got {type(x).__name__}.")
            raise TypeError(f"Expected a single Tensor input to head, but got {type(x).__name__}. If `required_features` has more than one feature, ensure that `fuse_module` is provided and returns a single Tensor.")

        if self._feature_adapter_supports_payload and payload is not None:
            x = self.feature_adapter(x, payload=payload, **feature_adapter_kwargs)
        else:
            x = self.feature_adapter(x, **feature_adapter_kwargs)

        if self._head_module_supports_payload and payload is not None:
            x = self.head_module(x, payload=payload, **head_module_kwargs)
        else:
            x = self.head_module(x, **head_module_kwargs)

        if x is None:
            raise RuntimeError(f"Active TaskHead {self.__class__.__name__} produced None output.")
        
        return x
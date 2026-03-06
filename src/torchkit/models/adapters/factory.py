from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from torchkit.models.adapters._feature_adapter import FeatureAdapter


@dataclass
class FeatureAdapterSpec:
    cls: type[FeatureAdapter] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class FeatureAdapterFactory:

    @staticmethod
    def build(
        spec: FeatureAdapterSpec,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> FeatureAdapter:
        """Builds a FeatureAdapter module. Optionally loads weights."""

        if spec.cls is None:
            raise ValueError("FeatureAdapterSpec.cls must be specified.")

        if not issubclass(spec.cls, FeatureAdapter):
            raise TypeError(
                f"FeatureAdapterSpec.cls must be a subclass of FeatureAdapter, got {spec.cls}."
            )

        if state_dict_path is not None and state_dict is not None:
            raise ValueError("Only one of state_dict_path or state_dict may be provided.")

        adapter = spec.cls(**spec.kwargs)

        if state_dict is not None:
            adapter.load_state_dict(state_dict, strict=strict)

        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            adapter.load_state_dict(loaded_state_dict, strict=strict)

        return adapter.to(device)
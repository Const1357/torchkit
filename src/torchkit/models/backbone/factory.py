from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from torchkit.models.backbone._backbone import Backbone


@dataclass
class BackboneSpec:
    cls: type[Backbone] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class BackboneFactory:

    @staticmethod
    def build(
        spec: BackboneSpec,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> Backbone:
        if spec.cls is None:
            raise ValueError("BackboneSpec.cls must be specified.")
        if not issubclass(spec.cls, Backbone):
            raise TypeError(f"BackboneSpec.cls must be a Backbone subclass, got {spec.cls}.")
        if state_dict_path is not None and state_dict is not None:
            raise ValueError("Only one of state_dict_path or state_dict may be provided.")

        backbone = spec.cls(**spec.kwargs)

        if state_dict is not None:
            backbone.load_state_dict(state_dict, strict=strict)
        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            backbone.load_state_dict(loaded_state_dict, strict=strict)

        return backbone.to(device)
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class HeadModuleSpec:
    cls: type[nn.Module] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class HeadModuleFactory:

    @staticmethod
    def build(
        spec: HeadModuleSpec,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> nn.Module:
        """Builds a head module. Optionally loads weights."""

        if spec.cls is None:
            raise ValueError("HeadModuleSpec.cls must be specified.")

        if not issubclass(spec.cls, nn.Module):
            raise TypeError(
                f"HeadModuleSpec.cls must be a subclass of torch.nn.Module, got {spec.cls}."
            )

        if state_dict_path is not None and state_dict is not None:
            raise ValueError("Only one of state_dict_path or state_dict may be provided.")

        head_module = spec.cls(**spec.kwargs)

        if state_dict is not None:
            head_module.load_state_dict(state_dict, strict=strict)

        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            head_module.load_state_dict(loaded_state_dict, strict=strict)

        return head_module.to(device)
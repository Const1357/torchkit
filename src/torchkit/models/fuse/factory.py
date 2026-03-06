from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from torchkit.models.fuse._fuse_module import FuseModule


@dataclass
class FuseModuleSpec:
    cls: type[FuseModule] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class FuseModuleFactory:

    @staticmethod
    def build(
        spec: FuseModuleSpec,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> FuseModule:
        """Builds a FuseModule. Optionally loads weights."""

        if spec.cls is None:
            raise ValueError("FuseModuleSpec.cls must be specified.")

        if not issubclass(spec.cls, FuseModule):
            raise TypeError(
                f"FuseModuleSpec.cls must be a subclass of FuseModule, got {spec.cls}."
            )

        if state_dict_path is not None and state_dict is not None:
            raise ValueError("Only one of state_dict_path or state_dict may be provided.")

        fuse_module = spec.cls(**spec.kwargs)

        if state_dict is not None:
            fuse_module.load_state_dict(state_dict, strict=strict)

        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            fuse_module.load_state_dict(loaded_state_dict, strict=strict)

        return fuse_module.to(device)
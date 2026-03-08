from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from torchkit.models.decision._decision_module import DecisionModule


@dataclass
class DecisionModuleSpec:
    cls: type[DecisionModule] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class DecisionModuleFactory:

    @staticmethod
    def build(
        spec: DecisionModuleSpec,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> DecisionModule:
        if spec.cls is None:
            raise ValueError("DecisionModuleSpec.cls must be specified.")

        if not issubclass(spec.cls, DecisionModule):
            raise TypeError(
                f"DecisionModuleSpec.cls must be a subclass of DecisionModule, got {spec.cls}."
            )

        if state_dict_path is not None and state_dict is not None:
            raise ValueError("Only one of state_dict_path or state_dict may be provided.")

        decision_module = spec.cls(**spec.kwargs)

        if state_dict is not None:
            decision_module.load_state_dict(state_dict, strict=strict)

        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            decision_module.load_state_dict(loaded_state_dict, strict=strict)

        return decision_module.to(device)
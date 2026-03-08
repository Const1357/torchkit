from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from torchkit.models.probability_mapping._probability_mapper import ProbabilityMapper


@dataclass
class ProbabilityMapperSpec:
    cls: type[ProbabilityMapper] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class ProbabilityMapperFactory:

    @staticmethod
    def build(
        spec: ProbabilityMapperSpec,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> ProbabilityMapper:
        if spec.cls is None:
            raise ValueError("ProbabilityMapperSpec.cls must be specified.")

        if not issubclass(spec.cls, ProbabilityMapper):
            raise TypeError(
                f"ProbabilityMapperSpec.cls must be a subclass of ProbabilityMapper, got {spec.cls}."
            )

        if state_dict_path is not None and state_dict is not None:
            raise ValueError("Only one of state_dict_path or state_dict may be provided.")

        probability_mapper = spec.cls(**spec.kwargs)

        if state_dict is not None:
            probability_mapper.load_state_dict(state_dict, strict=strict)

        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            probability_mapper.load_state_dict(loaded_state_dict, strict=strict)

        return probability_mapper.to(device)
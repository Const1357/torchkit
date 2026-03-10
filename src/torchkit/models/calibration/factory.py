from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from torchkit.models.calibration._calibrator import Calibrator


@dataclass
class CalibratorSpec:
    cls: type[Calibrator] | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    active: bool = True


class CalibratorFactory:

    @staticmethod
    def build(
        spec: CalibratorSpec,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> Calibrator:
        """Builds a calibrator module. Optionally loads weights."""

        if spec.cls is None:
            raise ValueError("CalibratorSpec.cls must be specified.")

        if not issubclass(spec.cls, Calibrator):
            raise TypeError(
                f"CalibratorSpec.cls must be a subclass of Calibrator, got {spec.cls}."
            )

        if state_dict_path is not None and state_dict is not None:
            raise ValueError(
                "Only one of state_dict_path or state_dict may be provided."
            )

        calibrator = spec.cls(active=spec.active, **spec.kwargs)

        if state_dict is not None:
            calibrator.load_state_dict(state_dict, strict=strict)

        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            calibrator.load_state_dict(loaded_state_dict, strict=strict)

        return calibrator.to(device)
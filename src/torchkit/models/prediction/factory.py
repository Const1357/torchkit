from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch

from torchkit.models.prediction._prediction_head import PredictionHead

from torchkit.models.calibration.factory import CalibratorSpec, CalibratorFactory
from torchkit.models.probability_mapping.factory import ProbabilityMapperSpec, ProbabilityMapperFactory
from torchkit.models.decision.factory import DecisionModuleSpec, DecisionModuleFactory


@dataclass
class PredictionHeadSpec:
    calibrator: Optional[CalibratorSpec] = None
    probability_mapper: Optional[ProbabilityMapperSpec] = None
    decision_module: Optional[DecisionModuleSpec] = None
    active: bool = True


class PredictionHeadFactory:

    @staticmethod
    def build(
        spec: PredictionHeadSpec,
        *,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,

        calibrator_state_dict_path: Optional[str] = None,
        calibrator_state_dict: Optional[dict[str, torch.Tensor]] = None,

        probability_mapper_state_dict_path: Optional[str] = None,
        probability_mapper_state_dict: Optional[dict[str, torch.Tensor]] = None,

        decision_module_state_dict_path: Optional[str] = None,
        decision_module_state_dict: Optional[dict[str, torch.Tensor]] = None,

        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> PredictionHead:
        if state_dict_path is not None and state_dict is not None:
            raise ValueError("Only one of state_dict_path or state_dict may be provided.")

        if (state_dict_path is not None or state_dict is not None) and any(
            x is not None
            for x in (
                calibrator_state_dict_path,
                calibrator_state_dict,
                probability_mapper_state_dict_path,
                probability_mapper_state_dict,
                decision_module_state_dict_path,
                decision_module_state_dict,
            )
        ):
            raise ValueError(
                "Whole PredictionHead loading (state_dict/state_dict_path) cannot be mixed with nested component state loading."
            )

        calibrator = None
        if spec.calibrator is not None:
            calibrator = CalibratorFactory.build(
                spec.calibrator,
                state_dict_path=calibrator_state_dict_path,
                state_dict=calibrator_state_dict,
                strict=strict,
                device=device,
            )

        probability_mapper = None
        if spec.probability_mapper is not None:
            probability_mapper = ProbabilityMapperFactory.build(
                spec.probability_mapper,
                state_dict_path=probability_mapper_state_dict_path,
                state_dict=probability_mapper_state_dict,
                strict=strict,
                device=device,
            )

        decision_module = None
        if spec.decision_module is not None:
            decision_module = DecisionModuleFactory.build(
                spec.decision_module,
                state_dict_path=decision_module_state_dict_path,
                state_dict=decision_module_state_dict,
                strict=strict,
                device=device,
            )

        prediction_head = PredictionHead(
            calibrator=calibrator,
            probability_mapper=probability_mapper,
            decision_module=decision_module,
            active=spec.active,
        )

        if state_dict is not None:
            prediction_head.load_state_dict(state_dict, strict=strict)
        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            prediction_head.load_state_dict(loaded_state_dict, strict=strict)

        return prediction_head.to(device)
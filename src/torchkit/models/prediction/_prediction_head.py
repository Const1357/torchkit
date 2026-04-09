from __future__ import annotations
from typing import Any, Optional

from torch import nn, Tensor

from torchkit.models.calibration._calibrator import Calibrator
from torchkit.models.decision._decision_module import DecisionModule
from torchkit.models.probability_mapping._probability_mapper import ProbabilityMapper


class PredictionHead(nn.Module):
    def __init__(
        self,
        *,
        calibrator: Optional[Calibrator] = None,
        probability_mapper: Optional[ProbabilityMapper] = None,
        decision_module: Optional[DecisionModule] = None,
        active: bool = True,
    ):
        super().__init__()

        self.calibrator = calibrator
        self.probability_mapper = probability_mapper
        self.decision_module = decision_module
        self._active = bool(active)

    @property
    def is_active(self) -> bool:
        return self._active
    
    @property
    def has_active_calibrator(self) -> bool:
        return self.calibrator is not None and self.calibrator.is_active

    @property
    def has_calibrator(self) -> bool:
        return self.calibrator is not None

    @property
    def has_trainable_decision_module(self) -> bool:
        return (
            self.decision_module is not None
            and getattr(self.decision_module, "is_trainable", False)
        )

    def enable(self) -> "PredictionHead":
        self._active = True
        return self

    def disable(self) -> "PredictionHead":
        self._active = False
        return self

    def forward(
        self,
        head_out: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self._active:
            return None

        if not isinstance(head_out, dict):
            raise TypeError(f"PredictionHead expected `head_out` to be a dict[str, Any], got {type(head_out).__name__}."
            )

        if "logits" not in head_out:
            raise KeyError(
                f"PredictionHead requires head output to contain 'logits'. "
                f"Available keys: {list(head_out.keys())}.")

        logits = head_out["logits"]
        if not isinstance(logits, Tensor):
            raise TypeError(f"PredictionHead expected head_out['logits'] to be a Tensor, got {type(logits).__name__}.")

        out = dict(head_out)

        calibrated_logits = logits
        if self.calibrator is not None and self.calibrator.is_active:
            calibrated_logits = self.calibrator(logits)
            out["calibrated_logits"] = calibrated_logits

        probs_in = calibrated_logits
        if self.probability_mapper is not None:
            probabilities = self.probability_mapper(probs_in)
            out["probabilities"] = probabilities
        else:
            probabilities = None

        if self.decision_module is not None:
            if probabilities is None:
                raise RuntimeError(
                    f"{self.__class__.__name__} cannot apply decision_module without probabilities. "
                    "Provide a probability_mapper."
                )
            predictions = self.decision_module(probabilities)
            out["predictions"] = predictions

        return out

    def to_spec(self):
        from torchkit.models.calibration.factory import CalibratorSpec
        from torchkit.models.decision.factory import DecisionModuleSpec
        from torchkit.models.prediction.factory import PredictionHeadSpec
        from torchkit.models.probability_mapping.factory import ProbabilityMapperSpec

        calibrator = None if self.calibrator is None else self.calibrator.to_spec()
        if calibrator is not None and not isinstance(calibrator, CalibratorSpec):
            raise TypeError(f"{self.calibrator.__class__.__name__}.to_spec() must return CalibratorSpec.")

        probability_mapper = (
            None if self.probability_mapper is None else self.probability_mapper.to_spec()
        )
        if probability_mapper is not None and not isinstance(probability_mapper, ProbabilityMapperSpec):
            raise TypeError(
                f"{self.probability_mapper.__class__.__name__}.to_spec() must return ProbabilityMapperSpec."
            )

        decision_module = None if self.decision_module is None else self.decision_module.to_spec()
        if decision_module is not None and not isinstance(decision_module, DecisionModuleSpec):
            raise TypeError(
                f"{self.decision_module.__class__.__name__}.to_spec() must return DecisionModuleSpec."
            )

        return PredictionHeadSpec(
            calibrator=calibrator,
            probability_mapper=probability_mapper,
            decision_module=decision_module,
            active=self.is_active,
        )

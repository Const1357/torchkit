from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch

from torchkit.models.Model import TorchkitModel

from torchkit.models.backbone.factory import BackboneSpec, BackboneFactory
from torchkit.models.head.factory import TaskHeadSpec, TaskHeadFactory
from torchkit.models.calibration.factory import CalibratorSpec, CalibratorFactory


@dataclass
class TorchkitModelSpec:
    backbone: BackboneSpec | None = None
    heads: dict[str, TaskHeadSpec] = field(default_factory=dict)
    calibrators: Optional[dict[str, CalibratorSpec]] = None


class TorchkitModelFactory:

    @staticmethod
    def build(
        spec: TorchkitModelSpec,
        *,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,

        backbone_state_dict_path: Optional[str] = None,
        backbone_state_dict: Optional[dict[str, torch.Tensor]] = None,

        head_state_dict_paths: Optional[dict[str, Optional[str]]] = None,
        head_state_dicts: Optional[dict[str, Optional[dict[str, torch.Tensor]]]] = None,

        head_component_state_dict_paths: Optional[dict[str, dict[str, Optional[str]]]] = None,
        head_component_state_dicts: Optional[dict[str, dict[str, Optional[dict[str, torch.Tensor]]]]] = None,

        calibrator_state_dict_paths: Optional[dict[str, Optional[str]]] = None,
        calibrator_state_dicts: Optional[dict[str, Optional[dict[str, torch.Tensor]]]] = None,

        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> TorchkitModel:
        if spec.backbone is None:
            raise ValueError("TorchkitModelSpec.backbone must be specified.")

        if not isinstance(spec.heads, dict) or len(spec.heads) == 0:
            raise ValueError("TorchkitModelSpec.heads must be a non-empty dict[str, TaskHeadSpec].")

        if not all(isinstance(k, str) and k for k in spec.heads.keys()):
            raise TypeError("All keys in TorchkitModelSpec.heads must be non-empty strings.")

        if spec.calibrators is not None:
            if not isinstance(spec.calibrators, dict):
                raise TypeError("TorchkitModelSpec.calibrators must be a dict[str, CalibratorSpec] or None.")
            if not all(isinstance(k, str) and k for k in spec.calibrators.keys()):
                raise TypeError("All keys in TorchkitModelSpec.calibrators must be non-empty strings.")

            extra = set(spec.calibrators.keys()) - set(spec.heads.keys())
            if extra:
                raise ValueError(
                    f"TorchkitModelSpec.calibrators contains keys not present in heads: {sorted(extra)}."
                )

        if state_dict_path is not None and state_dict is not None:
            raise ValueError("Only one of state_dict_path or state_dict may be provided.")

        if (state_dict_path is not None or state_dict is not None) and any(
            x is not None
            for x in (
                backbone_state_dict_path,
                backbone_state_dict,
                head_state_dict_paths,
                head_state_dicts,
                head_component_state_dict_paths,
                head_component_state_dicts,
                calibrator_state_dict_paths,
                calibrator_state_dicts,
            )
        ):
            raise ValueError(
                "Whole TorchkitModel loading (state_dict/state_dict_path) cannot be mixed with nested component state loading."
            )

        head_state_dict_paths = head_state_dict_paths or {}
        head_state_dicts = head_state_dicts or {}
        head_component_state_dict_paths = head_component_state_dict_paths or {}
        head_component_state_dicts = head_component_state_dicts or {}
        calibrator_state_dict_paths = calibrator_state_dict_paths or {}
        calibrator_state_dicts = calibrator_state_dicts or {}

        backbone = BackboneFactory.build(
            spec.backbone,
            state_dict_path=backbone_state_dict_path,
            state_dict=backbone_state_dict,
            strict=strict,
            device=device,
        )

        heads = {}
        for head_name, head_spec in spec.heads.items():
            head_sd_paths = head_component_state_dict_paths.get(head_name, {}) or {}
            head_sds = head_component_state_dicts.get(head_name, {}) or {}

            heads[head_name] = TaskHeadFactory.build(
                head_spec,
                state_dict_path=head_state_dict_paths.get(head_name),
                state_dict=head_state_dicts.get(head_name),

                fuse_state_dict_path=head_sd_paths.get("fuse_module"),
                fuse_state_dict=head_sds.get("fuse_module"),

                feature_adapter_state_dict_path=head_sd_paths.get("feature_adapter"),
                feature_adapter_state_dict=head_sds.get("feature_adapter"),

                head_module_state_dict_path=head_sd_paths.get("head_module"),
                head_module_state_dict=head_sds.get("head_module"),

                strict=strict,
                device=device,
            )

        calibrators = None
        if spec.calibrators is not None:
            calibrators = {}
            for head_name, cal_spec in spec.calibrators.items():
                calibrators[head_name] = CalibratorFactory.build(
                    cal_spec,
                    state_dict_path=calibrator_state_dict_paths.get(head_name),
                    state_dict=calibrator_state_dicts.get(head_name),
                    strict=strict,
                    device=device,
                )

        model = TorchkitModel(
            backbone=backbone,
            heads=heads,
            calibrators=calibrators,
        )

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=strict)
        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            model.load_state_dict(loaded_state_dict, strict=strict)

        return model.to(device)
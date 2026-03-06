from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Set, Tuple, List

import torch
from torch import nn

from torchkit.models.head._task_head import TaskHead
from torchkit.models.fuse.factory import FuseModuleSpec, FuseModuleFactory
from torchkit.models.adapters.factory import FeatureAdapterSpec, FeatureAdapterFactory
from torchkit.models.head_module.factory import HeadModuleSpec, HeadModuleFactory


@dataclass
class TaskHeadSpec:
    required_features: str | Set[str] | Tuple[str, ...] | List[str]

    fuse_module: Optional[FuseModuleSpec] = None
    feature_adapter: Optional[FeatureAdapterSpec] = None
    head_module: Optional[HeadModuleSpec] = None

    active: bool = True


class TaskHeadFactory:

    @staticmethod
    def build(
        spec: TaskHeadSpec,
        *,
        state_dict_path: Optional[str] = None,
        state_dict: Optional[dict[str, torch.Tensor]] = None,

        fuse_state_dict_path: Optional[str] = None,
        fuse_state_dict: Optional[dict[str, torch.Tensor]] = None,

        feature_adapter_state_dict_path: Optional[str] = None,
        feature_adapter_state_dict: Optional[dict[str, torch.Tensor]] = None,

        head_module_state_dict_path: Optional[str] = None,
        head_module_state_dict: Optional[dict[str, torch.Tensor]] = None,

        strict: bool = True,
        device: torch.device | str = "cpu",
    ) -> TaskHead:
        if state_dict_path is not None and state_dict is not None:
            raise ValueError("Only one of state_dict_path or state_dict may be provided.")

        if (state_dict_path is not None or state_dict is not None) and any(
            x is not None
            for x in (
                fuse_state_dict_path,
                fuse_state_dict,
                feature_adapter_state_dict_path,
                feature_adapter_state_dict,
                head_module_state_dict_path,
                head_module_state_dict,
            )
        ):
            raise ValueError(
                "Whole TaskHead loading (state_dict/state_dict_path) cannot be mixed with nested component state loading."
            )

        fuse_module: nn.Module | None = None
        if spec.fuse_module is not None:
            fuse_module = FuseModuleFactory.build(
                spec.fuse_module,
                state_dict_path=fuse_state_dict_path,
                state_dict=fuse_state_dict,
                strict=strict,
                device=device,
            )

        feature_adapter: nn.Module | None = None
        if spec.feature_adapter is not None:
            feature_adapter = FeatureAdapterFactory.build(
                spec.feature_adapter,
                state_dict_path=feature_adapter_state_dict_path,
                state_dict=feature_adapter_state_dict,
                strict=strict,
                device=device,
            )

        head_module: nn.Module | None = None
        if spec.head_module is not None:
            head_module = HeadModuleFactory.build(
                spec.head_module,
                state_dict_path=head_module_state_dict_path,
                state_dict=head_module_state_dict,
                strict=strict,
                device=device,
            )

        head = TaskHead(
            required_features=spec.required_features,
            fuse_module=fuse_module,
            feature_adapter=feature_adapter,
            head_module=head_module,
            active=spec.active,
        )

        if state_dict is not None:
            head.load_state_dict(state_dict, strict=strict)
        elif state_dict_path is not None:
            loaded_state_dict = torch.load(state_dict_path, map_location=device)
            head.load_state_dict(loaded_state_dict, strict=strict)

        return head.to(device)
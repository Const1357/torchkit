from __future__ import annotations
from typing import Any, Collection, Literal, Optional, ValuesView, ItemsView, KeysView

from torch import nn, Tensor

from torchkit.models.backbone._backbone import Backbone
from torchkit.models.head._task_head import TaskHead
from torchkit.models.prediction._prediction_head import PredictionHead

from warnings import warn


# For type hinting the overridden methods of nn.ModuleDict
class _TaskHeadModuleDict(nn.ModuleDict):
    def __getitem__(self, key: str) -> TaskHead:  # type: ignore[override]
        return super().__getitem__(key)  # runtime returns the module

    def items(self) -> ItemsView[str, TaskHead]:  # type: ignore[override]
        return super().items()

    def values(self) -> ValuesView[TaskHead]:  # type: ignore[override]
        return super().values()

    def keys(self) -> KeysView[str]:  # type: ignore[override]
        return super().keys()
    

class _PredictionHeadModuleDict(nn.ModuleDict):
    def __getitem__(self, key: str) -> PredictionHead:  # type: ignore[override]
        return super().__getitem__(key)  # runtime returns the module

    def items(self) -> ItemsView[str, PredictionHead]:  # type: ignore[override]
        return super().items()

    def values(self) -> ValuesView[PredictionHead]:  # type: ignore[override]
        return super().values()

    def keys(self) -> KeysView[str]:  # type: ignore[override]
        return super().keys()



class TorchkitModel(nn.Module):
    """Base class for all models. Defines the interface for a pure inference machine.\\
        Contracts:
        - All models must inherit from this class.
        - All models must have a backbone and one or more heads.
        - The backbone produces a dict of features.
        - Each head specifies which features it requires from the backbone, and produces an output for its task.
        - The model's forward method runs the backbone and all active heads, and returns a dict mapping head names to their outputs. You can optionally request backbone outputs.
        - The model is static: the backbone and heads are fixed at initialization and should not be changed after.
        - The model supports enabling/disabling and freezing/unfreezing heads, but does not support adding/removing heads or changing the backbone after initialization.
    """

    def __init__(
        self,
        backbone: Backbone,
        heads: dict[str, TaskHead],
        prediction_heads: Optional[dict[str, PredictionHead]] = None,
    ):
        """Prediction heads are attached to heads by name.
        If a prediction head is attached to a head, then the head MUST return a "logits" output."""
        super().__init__()

        if backbone is None:
            raise ValueError("`backbone` must be provided and non-None.")
        if heads is None:
            raise ValueError("`heads` must be provided and non-None.")

        # backbone
        if not isinstance(backbone, Backbone):
            raise TypeError(f"`backbone` must be an instance of Backbone, got {type(backbone)}.")
        self.backbone = backbone  # auto-registered

        # heads
        if not isinstance(heads, dict):
            raise TypeError(f"`heads` must be a dict mapping str to TaskHead, got {type(heads)}.")

        validated_heads = {}
        for name, head in heads.items():
            if not isinstance(name, str):
                raise TypeError(f"`heads` keys must be str, got {type(name)}: {name!r}.")
            if head is None:
                raise ValueError(f"`heads` values must be non-None, got None for key {name!r}.")
            if not isinstance(head, TaskHead):
                raise TypeError(
                    f"`heads` values must be instances of TaskHead, got {type(head)} for key {name!r}."
                )
            validated_heads[name] = head

        self.heads = _TaskHeadModuleDict(validated_heads)

        # prediction heads
        if prediction_heads is None:
            prediction_heads = {}
        if not isinstance(prediction_heads, dict):
            raise TypeError(f"`prediction_heads` must be a dict mapping str to PredictionHead, got {type(prediction_heads)}.")
        for name, prediction_head in prediction_heads.items():
            if not isinstance(name, str):
                raise TypeError(f"`prediction_heads` keys must be str, got {type(name)}: {name!r}.")
            if not isinstance(prediction_head, PredictionHead):
                raise TypeError(f"`prediction_heads` values must be instances of PredictionHead, got {type(prediction_head)} for key {name!r}.")
            if name not in heads.keys():
                raise ValueError(f"PredictionHead {name!r} does not have a corresponding head in `heads`: {set(heads.keys())}. Each prediction head must correspond to a head by name.")

        self.prediction_heads = _PredictionHeadModuleDict(prediction_heads)

    # Note: we do not cache these properties to avoid state/sync bugs.
    # Recomputing cost is negligible due to the small number of heads, and we can optimize later if needed.
    # If we do cache, we must enforce that the user only modifies properties via the provided API methods that update the state in a valid way.
    @property
    def head_names(self) -> set[str]:
        """Returns the set of all head names."""
        return set(self.heads.keys())
    @property
    def active_head_names(self) -> set[str]:
        """Returns the set of active head names."""
        return set(name for name, head in self.heads.items() if head.is_active)
    
    @property
    def prediction_head_names(self) -> set[str]:
        """Returns the set of all prediction head names."""
        return set(self.prediction_heads.keys())
    
    @property
    def active_prediction_head_names(self) -> set[str]:
        """Returns the set of active prediction head names."""
        if not self.prediction_heads:
            return set()
        return set(name for name in self.prediction_heads.keys() if self.prediction_heads[name].is_active)
    
    @property
    def all_required_features(self) -> set[str]:
        """Returns the set of all features required by all heads."""
        features = set()
        for head in self.heads.values():
            head: TaskHead
            features.update(head.required_features)
        return features
    
    @property
    def active_required_features(self) -> set[str]:
        """Returns the set of features required by the active heads."""
        features = set()
        for head in self.heads.values():
            head: TaskHead
            if head.is_active:
                features.update(head.required_features)
        return features
    
    @property
    def active_calibrator_names(self) -> set[str]:
        """Returns the names of active calibrators for active heads. Used for training calibrators."""
        names = set()
        for name, phead in self.prediction_heads.items():
            if self.heads[name].is_active and phead.has_active_calibrator:
                names.add(name)
        return names
    
    # API helpers for enabling/disabling heads by name, and freezing/unfreezing backbone and heads by name ---
    def enable_head(self, head_name: str | list[str] | set[str]) -> "TorchkitModel":
        """Enables the specified head(s) by name. Also enables the corresponding prediction head if it exists."""
        if isinstance(head_name, str):
            head_name = [head_name]
        if isinstance(head_name, set):
            head_name = list(head_name)
        for name in head_name:
            if name not in self.head_names:
                raise ValueError(f"Cannot enable head {name!r} because it does not exist in the model's heads: {self.head_names}.")
            if name not in self.active_head_names:
                self.heads[name].enable()
                try:
                    self.prediction_heads[name].enable()  # also enable corresponding prediction head if it exists
                except KeyError:
                    pass

        return self

    def disable_head(self, head_name: str | list[str] | set[str]) -> "TorchkitModel":
        """Disables the specified head(s) by name. Also disables the corresponding prediction head if it exists."""
        if isinstance(head_name, str):
            head_name = [head_name]
        if isinstance(head_name, set):
            head_name = list(head_name)
        for name in head_name:
            if name not in self.head_names:
                raise ValueError(f"Cannot disable head {name!r} because it does not exist in the model's heads: {self.head_names}.")
            if name in self.active_head_names:
                self.heads[name].disable()
                try:
                    self.prediction_heads[name].disable()  # also disable corresponding prediction head if it exists
                except KeyError:
                    pass
        return self

    def freeze_backbone(self) -> "TorchkitModel":
        self.backbone.freeze()
        return self
    def unfreeze_backbone(self) -> "TorchkitModel":
        self.backbone.unfreeze()
        return self

    def freeze_head(self, head_name: str | list[str] | set[str]) -> "TorchkitModel":
        """Freezes the specified head(s) by name."""
        if isinstance(head_name, str):
            head_name = [head_name]
        if isinstance(head_name, set):
            head_name = list(head_name)
        for name in head_name:
            if name not in self.head_names:
                raise ValueError(f"Cannot freeze head {name!r} because it does not exist in the model's heads: {self.head_names}.")
            self.heads[name].freeze()
        return self

    def unfreeze_head(self, head_name: str | list[str] | set[str]) -> "TorchkitModel":
        """Unfreezes the specified head(s) by name."""
        if isinstance(head_name, str):
            head_name = [head_name]
        if isinstance(head_name, set):
            head_name = list(head_name)
        for name in head_name:
            if name not in self.head_names:
                raise ValueError(f"Cannot unfreeze head {name!r} because it does not exist in the model's heads: {self.head_names}.")
            self.heads[name].unfreeze()
        return self
    
    def freeze_all_heads(self) -> "TorchkitModel":
        for head in self.heads.values():
            head.freeze()
        return self
    
    def unfreeze_all_heads(self) -> "TorchkitModel":
        for head in self.heads.values():
            head.unfreeze()
        return self
    
    @staticmethod
    def validate_load_path(path: str) -> None:
        import os
        if not isinstance(path, str) or not path.strip():
            raise TypeError(f"`path` must be a non-empty string, got {type(path)}.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Invalid path {path!r}: file does not exist.")

    def store(self, path: str) -> None:
        """Stores the model's state dict to the specified path.
        Overwrites if the file already exists.
        Creates parent directories if they do not exist."""
        import os
        if not isinstance(path, str) or not path.strip():
            raise TypeError(f"`path` must be a non-empty string, got {type(path)}.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        import torch
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: str = "cpu") -> None:
        """Loads the model's state dict from the specified path. Defaults to cpu."""
        self.validate_load_path(path)
        import torch
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)

    # Forward
    def forward(
        self,
        x: Tensor | dict[str, Tensor],
        *,
        backbone_kwargs: Optional[dict[str, Any]] = None,
        head_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        return_backbone_features: Optional[bool | str | Collection[str]] = None,
        ) -> dict[str, Any]:
        
        backbone_kwargs = backbone_kwargs or {}
        head_kwargs = head_kwargs or {}

        if isinstance(x, Tensor):
            payload: dict[str, Any] = {"x": x}  # wrap in dict for backbone
        elif isinstance(x, dict):
            payload: dict[str, Any] = x
        else:
            raise TypeError(f"Forward input must be a Tensor or dict[str, Tensor], got {type(x).__name__}.")
        
        payload = payload.copy()    # do not mutate input

        if "x" not in payload:
            raise KeyError("Payload dict must contain key 'x' (Tensor). Your backbone should support this key (or ignore it).")
        if not isinstance(payload["x"], Tensor):
            raise TypeError(f"payload['x'] must be a Tensor, got {type(payload['x']).__name__}.")
        
        bb_out: dict[str, Tensor] = self.backbone(payload, requested_features=self.active_required_features, **backbone_kwargs)
        
        out = {}
        if return_backbone_features is not None:
            # Namespace backbone features under a dedicated sub-dict
            to_return_backbone_out: dict[str, Tensor] = {}

            if isinstance(return_backbone_features, bool):
                if return_backbone_features:  # True means return all features
                    to_return_backbone_out.update(bb_out)

            elif isinstance(return_backbone_features, str):
                to_return_backbone_out[return_backbone_features] = bb_out[return_backbone_features]

            else:
                for feature in return_backbone_features:
                    to_return_backbone_out[feature] = bb_out[feature]

            if to_return_backbone_out:  # only insert if something was actually requested/returned
                out["backbone"] = to_return_backbone_out

        active_head_names = self.active_head_names  # cache for this forward pass 
        for head_name in active_head_names:
            
            head = self.heads[head_name]
            current_head_kwargs = head_kwargs.get(head_name, {})
            head_out = head(bb_out, payload=payload, head_module_kwargs=current_head_kwargs)

            if head_name in self.prediction_heads:
                if "logits" not in head_out:
                    raise KeyError(f"Head {head_name!r} has a PredictionHead but did not return 'logits' in its output.")
            
            out[head_name] = head_out

        return out
    
    def predict(
        self,
        x: Tensor | dict[str, Tensor],
        *task_names: str,
        backbone_kwargs: Optional[dict[str, Any]] = None,
        head_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        return_backbone_features: Optional[bool | str | Collection[str]] = None,
        return_raw_head_outputs: bool = False,
    ) -> dict[str, Any]:
        """Run prediction for the requested tasks.

        `task_names` temporarily overrides head active status by enabling the
        requested heads and disabling the rest for this call only.

        Return format:
        - keeps "backbone" if backbone features were requested
        - includes only requested task entries
        - for tasks with prediction heads, prediction outputs are attached under
        out[task_name]["predictions"]
        - for tasks without prediction heads, raw head outputs are returned
        """

        requested_task_names = set(task_names)
        if not requested_task_names:
            raise ValueError("At least one task name must be specified for prediction.")

        invalid = requested_task_names - self.head_names
        if invalid:
            raise ValueError(
                f"All task names must exist in the model. "
                f"Invalid names: {invalid}. Valid head names: {self.head_names}."
            )

        previous_active_heads = self.active_head_names.copy()

        self.enable_head(requested_task_names)
        self.disable_head(previous_active_heads - requested_task_names)

        try:
            fwd_out: dict[str, Any] = self(
                x,
                backbone_kwargs=backbone_kwargs,
                head_kwargs=head_kwargs,
                return_backbone_features=return_backbone_features,
            )

            out: dict[str, Any] = {}

            if "backbone" in fwd_out:
                out["backbone"] = fwd_out["backbone"]

            for task_name in requested_task_names:
                head_out = fwd_out[task_name]

                if not isinstance(head_out, dict):
                    raise TypeError(
                        f"Expected head output for task {task_name!r} to be a dict[str, Any], "
                        f"got {type(head_out).__name__}."
                    )

                if task_name in self.prediction_heads:
                    task_result = dict(head_out) if return_raw_head_outputs else {}
                    phead: PredictionHead = self.prediction_heads[task_name]

                    pred_out = phead(head_out=head_out)
                    if pred_out is None:
                        out[task_name] = task_result
                        continue
                    if not isinstance(pred_out, dict):
                        raise TypeError(
                            f"PredictionHead for task {task_name!r} must return dict[str, Any] or None, "
                            f"got {type(pred_out).__name__}."
                        )

                    task_result.update(pred_out)
                else:
                    # No prediction head: raw head output is the best available prediction surface
                    task_result = dict(head_out)

                out[task_name] = task_result

            return out

        finally:
            self.disable_head(self.head_names)
            self.enable_head(previous_active_heads)

    def to_spec(self):
        from torchkit.models.Model.factory import TorchkitModelSpec
        from torchkit.models.backbone.factory import BackboneSpec
        from torchkit.models.head.factory import TaskHeadSpec
        from torchkit.models.prediction.factory import PredictionHeadSpec

        backbone = self.backbone.to_spec()
        if not isinstance(backbone, BackboneSpec):
            raise TypeError(f"{self.backbone.__class__.__name__}.to_spec() must return BackboneSpec.")

        heads = {name: head.to_spec() for name, head in self.heads.items()}
        for name, head_spec in heads.items():
            if not isinstance(head_spec, TaskHeadSpec):
                raise TypeError(f"Head {name!r} to_spec() must return TaskHeadSpec.")

        prediction_heads = None
        if len(self.prediction_heads) > 0:
            prediction_heads = {
                name: prediction_head.to_spec()
                for name, prediction_head in self.prediction_heads.items()
            }
            for name, prediction_head_spec in prediction_heads.items():
                if not isinstance(prediction_head_spec, PredictionHeadSpec):
                    raise TypeError(
                        f"Prediction head {name!r} to_spec() must return PredictionHeadSpec."
                    )

        return TorchkitModelSpec(
            backbone=backbone,
            heads=heads,
            prediction_heads=prediction_heads,
        )

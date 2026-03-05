from __future__ import annotations
from typing import Any, Collection, Literal, Optional, ValuesView, ItemsView, KeysView

from torch import nn, Tensor

from torchkit.models.backbone._backbone import Backbone
from torchkit.models.calibrator._calibrator import Calibrator
from torchkit.models.task_head._task_head import TaskHead



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
    

class _CalibratorModuleDict(nn.ModuleDict):
    def __getitem__(self, key: str) -> Calibrator:  # type: ignore[override]
        return super().__getitem__(key)  # runtime returns the module

    def items(self) -> ItemsView[str, Calibrator]:  # type: ignore[override]
        return super().items()

    def values(self) -> ValuesView[Calibrator]:  # type: ignore[override]
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
        calibrators: Optional[dict[str, Calibrator]] = None,
    ):
        """calibrators are attached to heads by name.
        If a calibrator is attached to a head, then the head MUST return a "logits" output."""
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

        # calibrators
        if calibrators is None:
            calibrators = {}
        if not isinstance(calibrators, dict):
            raise TypeError(f"`calibrators` must be a dict mapping str to Calibrator, got {type(calibrators)}.")
        for name, calibrator in calibrators.items():
            if not isinstance(name, str):
                raise TypeError(f"`calibrators` keys must be str, got {type(name)}: {name!r}.")
            if not isinstance(calibrator, Calibrator):
                raise TypeError(f"`calibrators` values must be instances of Calibrator, got {type(calibrator)} for key {name!r}.")
            if name not in heads.keys():
                raise ValueError(f"Calibrator {name!r} does not have a corresponding head in `heads`: {set(heads.keys())}. Each calibrator must correspond to a head by name.")

        self.calibrators = _CalibratorModuleDict(calibrators)

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
    def calibrator_names(self) -> set[str]:
        """Returns the set of all calibrator names."""
        return set(self.calibrators.keys())
    
    @property
    def active_calibrator_names(self) -> set[str]:
        """Returns the names of calibrators for active heads."""
        return set(name for name in self.calibrators.keys() if self.heads[name].is_active)
    
    # API helpers for enabling/disabling heads by name, and freezing/unfreezing backbone and heads by name ---
    def enable_head(self, head_name: str | list[str]) -> "TorchkitModel":
        """Enables the specified head(s) by name."""
        if isinstance(head_name, str):
            head_name = [head_name]
        for name in head_name:
            if name not in self.head_names:
                raise ValueError(f"Cannot enable head {name!r} because it does not exist in the model's heads: {self.head_names}.")
            if name not in self.active_head_names:
                self.heads[name].enable()
        return self

    def disable_head(self, head_name: str | list[str]) -> "TorchkitModel":
        """Disables the specified head(s) by name."""
        if isinstance(head_name, str):
            head_name = [head_name]
        for name in head_name:
            if name not in self.head_names:
                raise ValueError(f"Cannot disable head {name!r} because it does not exist in the model's heads: {self.head_names}.")
            if name in self.active_head_names:
                self.heads[name].disable()
        return self

    def freeze_backbone(self) -> "TorchkitModel":
        self.backbone.freeze()
        return self
    def unfreeze_backbone(self) -> "TorchkitModel":
        self.backbone.unfreeze()
        return self

    def freeze_head(self, head_name: str | list[str]) -> "TorchkitModel":
        """Freezes the specified head(s) by name."""
        if isinstance(head_name, str):
            head_name = [head_name]
        for name in head_name:
            if name not in self.head_names:
                raise ValueError(f"Cannot freeze head {name!r} because it does not exist in the model's heads: {self.head_names}.")
            self.heads[name].freeze()
        return self
    
    def unfreeze_head(self, head_name: str | list[str]) -> "TorchkitModel":
        """Unfreezes the specified head(s) by name."""
        if isinstance(head_name, str):
            head_name = [head_name]
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
        head_kwargs: Optional[dict[str, dict[str, Any]]] = None,
        backbone_kwargs: Optional[dict[str, Any]] = None,
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

            if head_name in self.calibrators:
                if "logits" not in head_out:
                    raise KeyError(f"Head {head_name!r} has a calibrator but did not return 'logits' in its output. Calibrators require their head to return 'logits'.")
            
            out[head_name] = head_out

        return out

# ----------------------------------------------

# Example usage:

# def example():
#     # An MLP backbone that produces a single feature map "features".
#     # The feature map is then adopted by two heads: a classifier head and a regression head.
#     # The classifier head returns both logits and class probabilities.
#     # The regression head predicts 5 targets.


#     from torchkit.models.backbone.MLP_backbone import MLPBackbone
#     from torchkit.models.feature_adapter._feature_adapter import IdentityAdapter
#     from torchkit.models.head_module.classifier_head import ClassifierHeadMLP
#     from torchkit.models.head_module.regression_head import RegressorHeadMLP

#     from torchkit.models.task_head._task_head import TaskHead


#     # backbone
#     backbone = MLPBackbone(input_dim=64, hidden_dims=[128], output_dim=128)
#     print(backbone.available_features)  # {'features'}

#     # classification head
#     clf_head = TaskHead(

#         required_features={"features"}, # must be a subset of the backbone's available features
#         fuse_module=None,  # not needed for single feature, will be ignored
    
#         feature_adapter=IdentityAdapter(),

#         head_module=ClassifierHeadMLP(
#             input_dim=128,
#             hidden_dims=[64],
#             num_classes=10,
#             activation=nn.ReLU,
#             norm=nn.LayerNorm,
#             dropout=0.1),

#         active=True,
#     )

#     # regression head
#     reg_head = TaskHead(
#         required_features={"features"}, # must be a subset of the backbone's available features
#         fuse_module=None,  # not needed for single feature, will be ignored
    
#         feature_adapter=IdentityAdapter(),

#         head_module=RegressorHeadMLP(
#             input_dim=128,
#             hidden_dims=[64],
#             n_targets=5,
#             activation=nn.ReLU,
#             norm=nn.LayerNorm,
#             dropout=0.1),

#         active=True,
#     )

#     # model interface
#     model = TorchkitModel(backbone=backbone, heads={"clf": clf_head, "reg": reg_head}).to("cuda")
    
#     # input: batch of 32 samples, each with 64 features
#     import torch
#     x = torch.randn(32, 64, device="cuda")

#     # forward pass
#     out = model(
#         x,
#         return_backbone_features=True, 
#         head_kwargs={"clf": {"return_prob": True}}
#     )

#     print(out.keys())  # dict_keys(['backbone', 'clf', 'reg'])
#     print(out["backbone"].keys())  # dict_keys(['features'])
#     print(out["clf"].keys())  # dict_keys(['logits', 'prob'])
#     print(out["reg"].keys())  # dict_keys(['predictions'])

#     print(out["backbone"]["features"].shape)  # torch.Size([32, 128])
#     print(out["clf"]["logits"].shape)  # torch.Size([32, 10])
#     print(out["clf"]["prob"].shape)  # torch.Size([32, 10])
#     print(out["reg"]["predictions"].shape)  # torch.Size([32, 5])

#     print('--------------------')

# # example()

# def example2():
#     # Same as example(), but model input is a payload dict:
#     #   payload["x"]   -> image/features tensor
#     #   payload["tabular"] -> tabular tensor used by the fuser

#     import torch
#     from torch import nn

#     from torchkit.models.backbone.MLP_backbone import MLPBackbone
#     from torchkit.models._interface import TorchkitModel
#     from torchkit.models.task_head._task_head import TaskHead
#     from torchkit.models.feature_adapter._feature_adapter import IdentityAdapter
#     from torchkit.models.head_module.classifier_head import ClassifierHeadMLP
#     from torchkit.models.head_module.regression_head import RegressorHeadMLP
#     from torchkit.models.fuse_module._fuse_modules import TabularConcatFuseModule

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # ----------------------------
#     # backbone
#     # ----------------------------
#     backbone = MLPBackbone(input_dim=64, hidden_dims=[128], output_dim=128)
#     print(backbone.available_features)  # {'features'}

#     tab_dim = 10  # number of tabular features

#     # ----------------------------
#     # fuser: concat backbone feature with payload["tabular"]
#     # (structural only, no projections)
#     # ----------------------------
#     fuser = TabularConcatFuseModule(tabular_key="tabular", dim=1)

#     # ----------------------------
#     # heads
#     # NOTE: after fusion, feature dim becomes 128 + tab_dim
#     # ----------------------------
#     clf_head = TaskHead(
#         required_features={"features"},
#         fuse_module=fuser,  # <-- now fuse will be used even for a single feature
#         feature_adapter=IdentityAdapter(),
#         head_module=ClassifierHeadMLP(
#             input_dim=128 + tab_dim,
#             hidden_dims=[64],
#             num_classes=10,
#             activation=nn.ReLU,
#             norm=nn.LayerNorm,
#             dropout=0.1,
#         ),
#         active=True,
#     )

#     reg_head = TaskHead(
#         required_features={"features"},
#         fuse_module=fuser,
#         feature_adapter=IdentityAdapter(),
#         head_module=RegressorHeadMLP(
#             input_dim=128 + tab_dim,
#             hidden_dims=[64],
#             n_targets=5,
#             activation=nn.ReLU,
#             norm=nn.LayerNorm,
#             dropout=0.1,
#         ),
#         active=True,
#     )

#     model = TorchkitModel(backbone=backbone, heads={"clf": clf_head, "reg": reg_head}).to(device)

#     # batch
#     B = 32
#     batch = {
#         "x": torch.randn(B, 64, device=device),                     # main tensor
#         "tabular": torch.randn(B, tab_dim, device=device),          # aux tensor for fusion
#         "clf_labels" : torch.randint(0, 10, (B,), device=device),   # labels (not used in forward, just for illustration)
#         "reg_targets": torch.randn(B, 5, device=device),            # targets (not used in forward, just for illustration)
#     }

#     out = model(
#         batch,
#         return_backbone_features=True,
#         head_kwargs={"clf": {"return_prob": True}},
#     )

#     print(out.keys())  # dict_keys(['backbone', 'clf', 'reg'])
#     print(out["backbone"].keys())  # dict_keys(['features'])
#     print(out["backbone"]["features"].shape)  # torch.Size([32, 128])

#     print(out["clf"].keys())  # dict_keys(['logits', 'prob'])
#     print(out["reg"].keys())  # dict_keys(['predictions'])

#     print(out["clf"]["logits"].shape)        # torch.Size([32, 10])
#     print(out["clf"]["prob"].shape)          # torch.Size([32, 10])
#     print(out["reg"]["predictions"].shape)   # torch.Size([32, 5])

#     print('--------------------')


# example2()

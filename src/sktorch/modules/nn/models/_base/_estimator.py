from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
import os
from typing import Any, Dict, Mapping

import numpy as np
from torch import nn, Tensor
import torch


# helpers
from sktorch.random import set_seed
from sktorch._internal._util import _tag

from sktorch.modules.nn._util import _as_device

# TODO: checkpoints? are they finished?
# maybe continue after implementing `Trainer` (has optimizers, objectives, schedulers, models(backbone, adapter, head), etc)


# Checkpoint Format
@dataclass(frozen=True)
class CheckpointSpec:   # TODO: 
    """
    Defines what we store in a checkpoint.

    The key idea: you must be able to restore training *exactly*:
    - model weights
    - optimizer/scheduler/scaler state
    - RNG states
    - any counters (epoch/step)
    - init params/hparams to reconstruct the object if needed
    """
    format_version: int = 1



class SKTorchEstimatorBase(nn.Module, ABC):

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = _as_device(device)
        self.dtype = dtype
        self.to(self.device)

        self.is_fitted_: bool = False
        

    @abstractmethod
    def forward(
        self,
        X: Tensor,
        *,
        kwargs: Dict[str, Any] = None,
    ) -> Dict[str, Tensor | Any | None]:
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> "SKTorchEstimatorBase":
        """
        Base class does not implement training.

        To train the neural network wrap it in the `SKTorchTrainer` class.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.fit is not implemented. "
            "Use the `SKTorchTrainer` class and wrap your model."
        )
    
    # utilities
    def _to_tensor(self, X: np._ArrayLike) -> Tensor:
        return torch.from_numpy(np.asarray(X)).to(device=self.device, dtype=self.dtype)
    

    # checkpointing

    @classmethod
    def _init_signature_params(cls) -> set[str]:
        sig = inspect.signature(cls.__init__)
        names: set[str] = set()

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            names.add(name)
        return names



    def get_init_params(self) -> Dict[str, Any]:
        """Return a dictionary of the init parameters (no training params/state) and their values."""

        params: Dict[str, Any] = {}
        for name in self._init_signature_params():
            if not hasattr(self, name):
                continue
            v = getattr(self, name)

            # normalize torch types to serializable types
            if isinstance(v, torch.dtype):
                params[name] = str(v).replace("torch.", "")
            elif isinstance(v, torch.device):
                params[name] = str(v)
            else:
                params[name] = v

        return params
    
    # get_params and set_params are inherited from sklearn BaseEstimator


    def export_meta(self) -> Dict[str, Any]:
        """Export metadata about the model (not weights)."""
        meta: Dict[str, Any] = {}

        meta["is_fitted_"] = bool(self.is_fitted_)

        if self.classes_ is not None:
            meta["classes_"] = np.asarray(self.classes_)

        return meta
    
    def load_meta(self, meta: Mapping[str, Any]) -> None:
        """
        Restore metadata produced by export_meta().
        """
        if "is_fitted_" in meta:
            self.is_fitted_ = bool(meta["is_fitted_"])
        if "classes_" in meta and meta["classes_"] is not None:
            self.classes_ = np.asarray(meta["classes_"])
    #
    def estimator_state(self) -> Dict[str, Any]:

        return {
            "format" : {"name" : "sktorch-estimator", "version": 1},
            "class" : {"module": self.__class__.__module__, "name": self.__class__.__qualname__},
            "init_params" : self.get_init_params(),
            "model_state_dict" : self.state_dict(),
            "meta" : self.export_meta(),
        }
    
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.estimator_state(), path)

    def _import_by_name(module: str, name: str) -> Any:
        mod = __import__(module, fromlist=[name])
        obj = getattr(mod, name)
        if not isinstance(obj, type):
            raise TypeError(f"Imported object {name} from module {module} is not a class.")
        return obj
    
    @classmethod
    def load(cls,
        path: str, 
        *, 
        map_location: str | torch.device | None = None, 
        strict: bool = True
    ) -> "SKTorchEstimatorBase":
        
        ck: Dict[str, Any] = torch.load(path, map_location=map_location)
        fmt = ck.get("format", {})
        if fmt.get("name") != "sktorch-estimator":
            raise ValueError(f"Unrecognized estimator checkpoint format: {fmt}")
        
        meta = ck["class"]
        class_obj = cls._import_by_name(meta["module"], meta["name"])
        init_params = dict(ck.get("init_params", {}))

        # dtype string -> torch.dtype
        if "dtype" in init_params and isinstance(init_params["dtype"], str):
            init_params["dtype"] = getattr(torch, init_params["dtype"], torch.float32)
        
        model: SKTorchEstimatorBase = class_obj(**init_params)
        model.load_state_dict(ck["model_state_dict"], strict=strict)

        m = ck.get("meta", {})
        if isinstance(m, Mapping):
            model.load_meta(m)
        
        return model

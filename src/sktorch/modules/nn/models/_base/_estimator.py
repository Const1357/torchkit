from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Mapping

import numpy as np
from torch import nn, Tensor
import torch

from sklearn.base import BaseEstimator

from sktorch.modules.nn._util import _as_device

class SKTorchEstimatorBase(BaseEstimator, nn.Module, ABC):
    """
    Base class for sklearn-compatible PyTorch models.

    This class allows a PyTorch `nn.Module` to behave like a scikit-learn
    estimator while preserving full `state_dict` semantics.

    What this provides:
    + sklearn parameter introspection (`get_params`, `set_params`)
    + device and dtype management
    + consistent (de)serialization via `save()` / `load()`
    + persistence of fitted attributes (those ending with "_")

    Device & dtype:
    + The model is moved to the resolved device at initialization.
    + `_to_tensor()` converts NumPy-like inputs to tensors on the configured
      device and dtype.

    Fitting behavior:
    + `fit()` is NOT implemented at this level.
      Training should typically be handled by wrapping the model inside
      `SKTorchTrainer`.
    + After training, subclasses should set:
        `self.is_fitted_ = True`
      and may define additional fitted attributes (e.g., `classes_`,
      `n_outputs_`, etc.).

    Checkpoint format:
    Calling `save(path)` stores:
    + class metadata (module + class name)
    + constructor parameters (serialized)
    + model `state_dict`
    + fitted attributes

    Loading via `load(path)` reconstructs:
    + the estimator instance
    + constructor arguments
    + model weights
    + fitted state

    Subclass responsibilities:
    + Implement `forward(...)`.
    + Optionally implement `fit(...)` (or rely on `SKTorchTrainer`).
    + Extend `_fitted_state_keys()` if additional fitted attributes
      must persist across save/load.

    ---

    ### IMPORTANT — sklearn compatibility requirement

    **Subclasses MUST declare all constructor parameters explicitly in `__init__` and MUST NOT use `**kwargs` as a catch-all.**

    Example:

    ```python
    # Correct: (explicit parameters — sklearn compatible)
    def __init__(self, *, device=None, dtype=torch.float32, hidden_dim=128):
        ...
    # Incorrect: (catch-all **kwargs — NOT sklearn compatible)
    def __init__(self, **kwargs):
        ...
    ```
    """

    def __init__(
        self,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        nn.Module.__init__(self)
        BaseEstimator.__init__(self)

        self.device = device              # raw (str|device|None)
        self._device = _as_device(device) # resolved torch.device
        self.dtype = dtype
        self.to(self._device)

        self.is_fitted_: bool = False
        

    @abstractmethod
    def forward(self, X: Tensor, **kwargs: Any) -> Any:
        raise NotImplementedError
    
    # @abstractmethod   # removed the decorator to allow instantiation of interface classes that rely on SKTorchTrainer for fitting.
    # estimators remain pure inference machines, and training is handled by SKTorchTrainer.
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
        return torch.from_numpy(np.asarray(X)).to(device=self._device, dtype=self.dtype)


    def get_init_params(self) -> Dict[str, Any]:
        """
        Return constructor parameters for reconstruction (checkpoint-safe).

        We intentionally do NOT rely on sklearn's get_params(), because sklearn's
        parameter introspection can miss params in some multiple-inheritance /
        ABC setups, and we want checkpoints to be deterministic.

        Policy:
        - Read the most-derived __init__ signature (self.__class__.__init__).
        - Collect all explicitly declared parameters (excluding *args/**kwargs).
        - Fetch each value from the instance attribute of the same name.
        """
        import inspect

        sig = inspect.signature(self.__class__.__init__)
        out: Dict[str, Any] = {}

        for name, p in sig.parameters.items():
            if name == "self":
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                # ignore *args/**kwargs
                continue

            if not hasattr(self, name):
                raise AttributeError(
                    f"{self.__class__.__name__}.get_init_params(): "
                    f"__init__ declares parameter '{name}', but the instance has no attribute '{name}'. "
                    f"Store sklearn params as attributes with the same name (e.g., self.{name} = {name})."
                )

            out[name] = getattr(self, name)

        return out
    

    def _fitted_state_keys(self) -> tuple[str, ...]:
        """
        Names of fitted attributes to persist across save/load.
        Sklearn convention: fitted attributes end with '_'.
        """
        return ("is_fitted_",)
    
    # EXAMPLE: if a classifier has `classes_` after fitting, include that in the fitted state:
    # def _fitted_state_keys(self):
    #     return super()._fitted_state_keys() + ("classes_",)


    def _get_fitted_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        for k in self._fitted_state_keys():
            if hasattr(self, k):
                state[k] = getattr(self, k)
        return state


    def _set_fitted_state(self, state: Mapping[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)

    # (de)serialization helpers
    @classmethod
    def _serialize_params(cls, obj: Any) -> Any:

        # factories / custom serializables: *MUST* implement `to_dict()` for custom serialization, and `from_dict()` for deserialization
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            d = to_dict()

            if not isinstance(d, dict):
                raise TypeError(f"{type(obj).__name__}.to_dict() must return a dict, got {type(d)}.")

            if "__type__" not in d:
                raise ValueError(f"{type(obj).__name__}.to_dict() must include '__type__' for deserialization.")

            return {k: cls._serialize_params(v) for k, v in d.items()}

        # containers
        if isinstance(obj, dict):
            return {k: cls._serialize_params(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cls._serialize_params(v) for v in obj]

        # torch types -> stable primitives
        if isinstance(obj, torch.dtype):
            return {"__type__": "torch.dtype", "value": str(obj).replace("torch.", "")}
        if isinstance(obj, torch.device):
            return {"__type__": "torch.device", "value": str(obj)}

        return obj
    
    @classmethod
    def _deserialize_params(cls, obj: Any) -> Any:

        if isinstance(obj, dict):
            
            # torch special types
            t = obj.get("__type__")
            if t == "torch.dtype":
                return getattr(torch, obj.get("value", "float32"), torch.float32)
            if t == "torch.device":
                return torch.device(obj["value"])

            # factory types (discriminator preferred)
            if "__type__" in obj:
                typ = obj["__type__"]
                dd = {k: cls._deserialize_params(v) for k, v in obj.items()}

                if typ == "ModuleFactory":
                    from sktorch.modules.nn.models.factory import ModuleFactory
                    return ModuleFactory.from_dict(dd)

                if typ == "AdapterFactory":
                    from sktorch.modules.nn.FeatureAdapters import AdapterFactory
                    return AdapterFactory.from_dict(dd)
                
                return dd  # unknown tagged dict

            # plain dict
            return {k: cls._deserialize_params(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [cls._deserialize_params(v) for v in obj]

        return obj


    def estimator_state(self) -> Dict[str, Any]:

        return {
            "format" : {"name" : "sktorch-estimator", "version": 1},
            "class" : {"module": self.__class__.__module__, "name": self.__class__.__qualname__},
            "init_params" : self._serialize_params(self.get_init_params()),
            "model_state_dict" : self.state_dict(),
            "fitted_state": self._get_fitted_state(),
        }
    
    def store_estimator(self, path: str) -> None:
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        torch.save(self.estimator_state(), path)

    @staticmethod
    def _import_by_name(module: str, name: str) -> Any:
        mod = __import__(module, fromlist=["*"])
        obj: Any = mod
        for part in name.split("."):
            obj = getattr(obj, part)
        if not isinstance(obj, type):
            raise TypeError(f"Imported object {name} from module {module} is not a class, got {type(obj)}.")
        return obj

    @classmethod
    def load_estimator(
        cls,
        path: str,
        *,
        map_location: str | torch.device | None = None,
        strict: bool = False,  # lazy loading => allow missing keys by default, but user can override with strict=True
    ) -> "SKTorchEstimatorBase":
        ck: Dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)

        fmt = ck.get("format", {})
        if fmt.get("name") != "sktorch-estimator":
            raise ValueError(f"Unrecognized estimator checkpoint format: {fmt}")

        meta = ck["class"]
        class_obj = cls._import_by_name(meta["module"], meta["name"])

        cls_obj_deserialize = getattr(class_obj, "_deserialize_params", None)
        if not callable(cls_obj_deserialize):
            raise TypeError(f"Loaded estimator class {class_obj.__module__}.{class_obj.__qualname__} must implement _deserialize_params() for checkpoint loading.")

        # decode using the class we are about to instantiate
        init_params = cls_obj_deserialize(ck.get("init_params", {}))
        if not isinstance(init_params, dict):
            raise TypeError(f"Checkpoint init_params must decode to dict, got {type(init_params)}.")

        model: SKTorchEstimatorBase = class_obj(**init_params)
        model.load_state_dict(ck["model_state_dict"], strict=strict)

        fs = ck.get("fitted_state", {})
        if isinstance(fs, Mapping):
            model._set_fitted_state(fs)

        return model

    # aliases
    def save_estimator(self, path: str) -> None:
        self.store_estimator(path)

    def save(self, path: str) -> None:
        self.store_estimator(path)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device | None = None, strict: bool = False) -> "SKTorchEstimatorBase":
        return cls.load_estimator(path, map_location=map_location, strict=strict)
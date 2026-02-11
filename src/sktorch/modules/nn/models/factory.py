from dataclasses import dataclass, field
from typing import Dict, Any, Type

from sktorch.modules.nn._util import _cls_to_path, _import_from_path

from torch import nn, Tensor


@dataclass(frozen=True)
class ModuleFactory:
    """
    ModuleFactory specifies how to construct a torch.nn.Module, either from a class path and kwargs, or from a type and kwargs.
    It supports two main methods of construction:
    1. build(): constructs the module directly from the stored cls_path and kwargs.
    2. from_input(dummy): constructs the module by inferring necessary parameters (like input shape) from a dummy input tensor. This requires that the target class's __init__ method accepts either a 'dummy' argument or an 'input_shape' argument.
    3. from_dict(): constructs the module from a dictionary of parameters, useful for deserialization.
    This factory should be used for the creation of all backbones and heads.
    """

    cls_path: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_type(cls, t: Type[Any], **kwargs: Any) -> "ModuleFactory":
        return cls(cls_path=_cls_to_path(t), kwargs=dict(kwargs))
    
    def build(self, **runtime_build_args:Any) -> nn.Module:
        cls = _import_from_path(self.cls_path)
        module = cls(**self.kwargs, **runtime_build_args)
        if not isinstance(module, nn.Module):
            raise TypeError(f"Factory {self.cls_path}: Built object is not a torch.nn.Module, got {type(module)}.")
        return module
    
    def from_input(self, dummy: Tensor) -> nn.Module:
        """Assumes the class can be constructed with an input_shape argument that is inferred from the dummy input."""

        if dummy.ndim < 2:
            raise ValueError(f"Dummy input must be at least 2D (batch dimension + feature dimensions), got shape {tuple(dummy.shape)}.")
        
        cls = _import_from_path(self.cls_path)
        import inspect
        sig = inspect.signature(cls.__init__)

        if "dummy" in sig.parameters:
            return self.build(dummy=dummy)
        elif "input_shape" in sig.parameters:
            input_shape = tuple(dummy.shape[1:]) # exclude batch dimension
            return self.build(input_shape=input_shape)
        else:
            raise ValueError(f"Factory {self.cls_path}: Class {cls}.__init__ does not accept 'dummy' or 'input_shape', cannot use ModuleFactory.from_input().")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the factory to a dictionary for serialization."""
        return {
            "__type__": "ModuleFactory",
            "cls_path": self.cls_path,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModuleFactory":
        return cls(cls_path=d["cls_path"], kwargs=dict(d.get("kwargs", {})))
    


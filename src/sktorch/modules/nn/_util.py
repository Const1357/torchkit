from typing import Any, Type
import torch



def _as_device(device: str | torch.device | None) -> torch.device | None:
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    raise TypeError(f"Invalid device type: {type(device)}")

def _cls_to_path(cls: Type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"

def _import_from_path(path: str) -> Any:
    """
    path format: "some.module:QualName.Inner"
    """
    if ":" not in path:
        raise ValueError(f"Invalid class path {path!r}. Expected 'module:QualName'.")
    module_name, qualname = path.split(":", 1)
    mod = __import__(module_name, fromlist=["*"])
    obj: Any = mod
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise TypeError(f"Imported object {qualname} from module {module_name} is not a class.")
    return obj
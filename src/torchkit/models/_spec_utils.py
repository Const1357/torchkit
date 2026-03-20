from __future__ import annotations

import inspect
from typing import Any


def clone_spec_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: clone_spec_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_spec_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_spec_value(v) for v in value)
    if isinstance(value, set):
        return {clone_spec_value(v) for v in value}
    return value


def normalize_spec_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: clone_spec_value(value) for key, value in kwargs.items()}


def resolve_spec_kwargs(instance: object) -> dict[str, Any]:
    kwargs = getattr(instance, "_spec_kwargs", None)
    if kwargs is not None:
        return normalize_spec_kwargs(kwargs)

    signature = inspect.signature(instance.__class__.__init__)
    required_params = [
        param.name
        for param in list(signature.parameters.values())[1:]
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and param.default is inspect.Parameter.empty
    ]

    if required_params:
        raise ValueError(
            f"{instance.__class__.__name__} cannot be converted to a spec automatically because "
            f"its constructor kwargs were not recorded and it requires {required_params}."
        )

    return {}

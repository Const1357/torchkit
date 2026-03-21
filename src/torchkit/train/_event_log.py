from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import json
import os
import uuid

import torch


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


def _jsonify(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    if isinstance(x, set):
        return sorted(_jsonify(v) for v in x)
    if torch.is_tensor(x):
        return {
            "__kind__": "tensor_summary",
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "device": str(x.device),
        }
    if is_dataclass(x):
        return _jsonify(asdict(x))
    return repr(x)


def default_log_dir(*, prefix: str, base_dir: Optional[str] = None) -> str:
    if base_dir is None:
        root = os.path.join(os.getcwd(), "torchkit_logs")
    else:
        root = base_dir
    path = os.path.join(root, f"{prefix}_{_run_id()}")
    os.makedirs(path, exist_ok=True)
    return path


class JsonlEventLogger:
    def __init__(
        self,
        path: str,
        *,
        scope: str,
        echo_console: bool = False,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        self.path = path
        self.scope = scope
        self.echo_console = bool(echo_console)
        self.context = dict(context or {})

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def child(
        self,
        path: str,
        *,
        scope: str,
        context: Optional[dict[str, Any]] = None,
    ) -> "JsonlEventLogger":
        merged = dict(self.context)
        merged.update(context or {})
        return JsonlEventLogger(
            path,
            scope=scope,
            echo_console=self.echo_console,
            context=merged,
        )

    def emit(
        self,
        event: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        message: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        merged_context = dict(self.context)
        merged_context.update(context or {})

        entry = {
            "timestamp": _timestamp_utc(),
            "event": event,
            "scope": self.scope,
            "message": message,
            "context": _jsonify(merged_context),
            "payload": _jsonify(payload or {}),
            "log_file": self.path,
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        if self.echo_console:
            if message:
                print(message)
            else:
                print(f"[{self.scope}] {event}")

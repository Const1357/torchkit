from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedConfig:
    enabled: bool = False
    backend: str = "nccl"
    init_method: str | None = None
    timeout_s: int = 1800
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, got {self.timeout_s}.")

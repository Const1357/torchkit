from __future__ import annotations

from dataclasses import dataclass
import os

import torch


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


@dataclass(frozen=True)
class DistributedContext:
    global_rank: int
    world_size: int
    local_rank: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.global_rank == 0

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    @classmethod
    def from_env(cls) -> "DistributedContext":
        return cls(
            global_rank=_get_env_int("RANK", 0),
            world_size=_get_env_int("WORLD_SIZE", 1),
            local_rank=_get_env_int("LOCAL_RANK", 0),
        )

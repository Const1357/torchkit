from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional
import copy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torchkit.distributed.config import DistributedConfig
from torchkit.distributed.context import DistributedContext


@dataclass
class DDPStrategy:
    config: DistributedConfig
    context: DistributedContext
    process_group: Optional[dist.ProcessGroup] = None

    def __post_init__(self) -> None:
        self._owns_process_group = False

    def __deepcopy__(self, memo):
        copied = type(self)(
            config=copy.deepcopy(self.config, memo),
            context=copy.deepcopy(self.context, memo),
            process_group=None,
        )
        copied._owns_process_group = False
        return copied

    @property
    def is_enabled(self) -> bool:
        return bool(self.config.enabled and self.context.world_size > 1)

    @property
    def is_main_process(self) -> bool:
        return self.context.is_main_process

    @property
    def device(self) -> torch.device:
        return self.context.device

    def initialize(self) -> None:
        if not self.is_enabled:
            return
        if self.process_group is not None:
            return
        if self.context.device.type == "cuda":
            # Bind each rank to its assigned GPU before NCCL initialization.
            torch.cuda.set_device(self.context.local_rank)
        if not dist.is_initialized():
            kwargs: dict[str, Any] = {
                "backend": self.config.backend,
                "rank": self.context.global_rank,
                "world_size": self.context.world_size,
                "timeout": timedelta(seconds=int(self.config.timeout_s)),
            }
            if self.config.init_method is not None:
                kwargs["init_method"] = self.config.init_method
            dist.init_process_group(**kwargs)
            self._owns_process_group = True
        self.process_group = dist.group.WORLD

    def finalize(self) -> None:
        if not self.is_enabled:
            return
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
            self.process_group = None

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if not self.is_enabled:
            return model
        try:
            module_device = next(model.parameters()).device
        except StopIteration:
            module_device = self.context.device
        if module_device.type == "cuda":
            return DistributedDataParallel(
                model,
                device_ids=[self.context.local_rank],
                output_device=self.context.local_rank,
                process_group=self.process_group,
                find_unused_parameters=self.config.find_unused_parameters,
                broadcast_buffers=self.config.broadcast_buffers,
            )
        return DistributedDataParallel(
            model,
            process_group=self.process_group,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers,
        )

    def set_epoch(self, loader: Any, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def barrier(self) -> None:
        if not self.is_enabled:
            return
        dist.barrier(group=self.process_group)

    def _collective_device(self) -> torch.device:
        if self.config.backend.lower() == "nccl":
            return self.context.device
        return torch.device("cpu")

    def all_reduce_sum_float(self, value: float) -> float:
        if not self.is_enabled:
            return float(value)
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self._collective_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group)
        return float(tensor.cpu().item())

    def all_reduce_sum_int(self, value: int) -> int:
        if not self.is_enabled:
            return int(value)
        tensor = torch.tensor(int(value), dtype=torch.int64, device=self._collective_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group)
        return int(tensor.cpu().item())

    def all_gather_object(self, obj: Any) -> list[Any]:
        if not self.is_enabled:
            return [obj]
        gathered = [None for _ in range(self.context.world_size)]
        dist.all_gather_object(gathered, obj, group=self.process_group)
        return gathered

    def broadcast_object(self, obj: Any, *, src: int = 0) -> Any:
        if not self.is_enabled:
            return obj
        payload = [obj]
        dist.broadcast_object_list(payload, src=src, group=self.process_group)
        return payload[0]

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Type

import torch
import torch.distributed as dist
from torch.optim import Optimizer

# 25 MB per broadcast bucket — balances latency vs pipeline overlap
_BUCKET_BYTES = 25 * 1024 * 1024


class ShardedOptimizer(Optimizer):
    """ZeRO-1 style optimizer: shard optimizer state across ranks.

    Every rank holds the full model parameters and computes gradients on all
    of them, but each rank's inner optimizer only manages a contiguous shard.
    After ``step()``, updated shards are broadcast so every rank sees the
    same parameters.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        optimizer_cls: Type[Optimizer],
        **kwargs: Any,
    ) -> None:

        if not dist.is_initialized():
            raise RuntimeError("Must call dist.init_process_group before")

        self.all_params = list(params)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        # Round-robin assignment so large/small params are spread evenly.
        # _param_owner[i] = which rank "owns" (optimizes) param i.
        self._param_owner: list[int] = [
            i % self.world_size for i in range(len(self.all_params))
        ]

        # Group params by owner rank
        rank_params: list[list[torch.nn.Parameter]] = [[] for _ in range(self.world_size)]
        for i, p in enumerate(self.all_params):
            rank_params[self._param_owner[i]].append(p)

        # Pre-compute ~25MB broadcast buckets per owner rank.
        # Each bucket is (owner_rank, [params_in_bucket]).
        self._buckets: list[tuple[int, list[torch.nn.Parameter]]] = []
        for owner in range(self.world_size):
            cur_bucket: list[torch.nn.Parameter] = []
            cur_bytes = 0
            for p in rank_params[owner]:
                cur_bucket.append(p)
                cur_bytes += p.numel() * p.element_size()
                if cur_bytes >= _BUCKET_BYTES:
                    self._buckets.append((owner, cur_bucket))
                    cur_bucket = []
                    cur_bytes = 0
            if cur_bucket:
                self._buckets.append((owner, cur_bucket))

        # Only create an inner optimizer for this rank's shard
        local_params = rank_params[self.rank]
        if local_params:
            self.inner_optimizer = optimizer_cls(local_params, **kwargs)
        else:
            self.inner_optimizer = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        # Must zero ALL params (not just local shard), because backward()
        # computes gradients on all parameters regardless of ownership.
        for p in self.all_params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        # Each rank only updates its own shard
        loss = None
        if self.inner_optimizer is not None:
            loss = self.inner_optimizer.step(closure)

        # Launch all bucket broadcasts async, then wait + unpack
        pending: list[tuple[int, torch.Tensor, dist.Work, list[torch.nn.Parameter]]] = []
        for owner, bucket_params in self._buckets:
            flat = torch.cat([p.data.reshape(-1) for p in bucket_params])
            handle = dist.broadcast(flat, src=owner, async_op=True)
            pending.append((owner, flat, handle, bucket_params))

        for owner, flat, handle, bucket_params in pending:
            handle.wait()
            if owner != self.rank:
                offset = 0
                for p in bucket_params:
                    p.data.copy_(flat[offset : offset + p.numel()].reshape(p.shape))
                    offset += p.numel()

        return loss

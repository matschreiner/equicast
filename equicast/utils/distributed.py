"""Node-sharding primitives for distributed PaiNN training.

Each GPU holds a contiguous slice of the hidden node tensor. During message
passing, every rank all-gathers the full hidden tensor to index source features,
then scatters only into its local destination range. Gradients rely on DDP's
all-reduce to assemble the full encoder/decoder parameter gradients; no extra
all-gather is needed in the backward pass.
"""

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup


class _GatherParallelSection(torch.autograd.Function):
    """Forward: all_gather (local shard → full). Backward: slice to local rank."""

    @staticmethod
    def forward(ctx, x: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        world_size = dist.get_world_size(group)
        if world_size == 1:
            ctx.world_size = 1
            return x
        ctx.world_size = world_size
        ctx.local_size = x.size(0)
        out = x.new_empty(world_size * x.size(0), *x.shape[1:])
        dist.all_gather_into_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad: Tensor):
        if ctx.world_size == 1:
            return grad, None
        rank = dist.get_rank(ctx.group)
        start = rank * ctx.local_size
        return grad[start : start + ctx.local_size].contiguous(), None


class _ShardParallelSection(torch.autograd.Function):
    """Forward: slice to local rank. Backward: sparse full tensor (zeros outside local range).

    DDP's all-reduce assembles the correct full gradient from all ranks.
    """

    @staticmethod
    def forward(ctx, x: Tensor, group: ProcessGroup) -> Tensor:
        ctx.group = group
        world_size = dist.get_world_size(group)
        if world_size == 1:
            ctx.world_size = 1
            return x
        assert x.size(0) % world_size == 0, (
            f"Hidden node dim {x.size(0)} must be divisible by world_size {world_size}"
        )
        ctx.world_size = world_size
        ctx.total_size = x.size(0)
        ctx.local_size = x.size(0) // world_size
        rank = dist.get_rank(group)
        start = rank * ctx.local_size
        return x[start : start + ctx.local_size].contiguous()

    @staticmethod
    def backward(ctx, grad: Tensor):
        if ctx.world_size == 1:
            return grad, None
        rank = dist.get_rank(ctx.group)
        full = grad.new_zeros(ctx.total_size, *grad.shape[1:])
        start = rank * ctx.local_size
        full[start : start + ctx.local_size] = grad
        return full, None


def gather_tensor(x: Tensor, group: ProcessGroup) -> Tensor:
    """All-gather sharded tensor along dim 0. All shards must have equal size."""
    return _GatherParallelSection.apply(x, group)


def shard_tensor(x: Tensor, group: ProcessGroup) -> Tensor:
    """Slice tensor to local rank's shard along dim 0. Requires equal division."""
    return _ShardParallelSection.apply(x, group)

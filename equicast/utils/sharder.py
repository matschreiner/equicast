"""NodeSharder: generic distributed primitive for node sharding.

Exposes shard / gather / run_block(block, scalar, vector, edge_type, edges).
"""

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ProcessGroup

from equicast.utils.distributed import gather_tensor, shard_tensor


class NodeSharder:
    """Shards nodes across GPUs.

    Requires N % world_size == 0 for any node set being sharded.
    """

    @classmethod
    def create(cls) -> "NodeSharder | None":
        """Return a NodeSharder for the current world group, or None if not distributed."""
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return None
        return cls(dist.group.WORLD)

    def __init__(self, group: ProcessGroup):
        self.group = group

    @property
    def rank(self) -> int:
        return dist.get_rank(self.group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.group)

    def shard(self, scalar: torch.Tensor, vector: torch.Tensor):
        """Split [N, H] / [N, H, 2] → local [N/P, H] / [N/P, H, 2]."""
        return shard_tensor(scalar, self.group), shard_tensor(vector, self.group)

    def gather(self, scalar: torch.Tensor, vector: torch.Tensor):
        """All-gather [N/P, H] / [N/P, H, 2] → [N, H] / [N, H, 2]."""
        return gather_tensor(scalar, self.group), gather_tensor(vector, self.group)

    def run_block(self, block, scalar, vector, edge_type, edges):
        """Run one PaiNNBlock with node sharding. Dispatch via edge_type.

        encoder (dst=hidden):
            scalar / vector are the full data tensors [N_data, H]. Shards hidden
            dst internally; returns [N_h/P, H].

        h2h (src=hidden, dst=hidden):
            scalar / vector are the local hidden shard [N_h/P, H]. All-gathers
            for src; returns [N_h/P, H].

        decoder (dst=data):
            scalar / vector are the local hidden shard [N_h/P, H]. Gathers
            hidden for src, shards data dst, gathers result; returns [N_data, H].
        """
        src_type, _, dst_type = edge_type

        if dst_type == "hidden":
            N_dst = int(edges["edge_index"][1].max().item()) + 1
            if src_type == "data":
                # encoder: scalar is full [N_data, H]; shard hidden internally
                scalar_local, vector_local = self.shard(scalar[:N_dst], vector[:N_dst])
                return self._run(block, scalar_local, vector_local, edges,
                                 src_scalar=scalar, src_vector=vector)
            else:
                # h2h: scalar is already the hidden shard; all-gather for src
                return self._run(block, scalar, vector, edges)

        if dst_type == "data":
            N_dst = int(edges["edge_index"][1].max().item()) + 1
            src_s, src_v = self.gather(scalar, vector)
            local_size = N_dst // self.world_size
            local_start = self.rank * local_size
            scalar_local = src_s.new_zeros(local_size, src_s.size(-1))
            vector_local = src_v.new_zeros(local_size, src_v.size(-2), 2)
            return self._run(block, scalar_local, vector_local, edges,
                             src_scalar=src_s, src_vector=src_v,
                             local_start=local_start)

        raise ValueError(f"Unsupported edge type for sharding: {edge_type}")

    def _run(self, block, scalar_local, vector_local, edges,
             src_scalar=None, src_vector=None, local_start=None):
        local_size = scalar_local.size(0)
        if local_start is None:
            local_start = self.rank * local_size
        if src_scalar is None:
            src_scalar, src_vector = self.gather(scalar_local, vector_local)

        src, dst = edges["edge_index"].long()
        mask = (dst >= local_start) & (dst < local_start + local_size)
        local_edge_index = torch.stack([src[mask], dst[mask] - local_start])
        local_dirs = edges.edge_dirs[mask]
        edge_emb = block.embed_edge_length(edges.edge_length[mask])

        d_scalar, d_vector = block.message_passing(
            scalar_local, vector_local, local_edge_index, local_dirs, edge_emb,
            src_scalar=src_scalar, src_vector=src_vector,
        )
        scalar_local, vector_local = block.norm1(scalar_local + d_scalar, vector_local + d_vector)
        d_scalar, d_vector = block.update(scalar_local, vector_local)
        return block.norm2(scalar_local + d_scalar, vector_local + d_vector)


"""PaiNN backbone model."""

import warnings

import torch
from torch import nn
from torch_geometric.utils import scatter

from equicast.model.layers.equivariant_conv import EquivariantLinear
from equicast.model.layers.mlp import MLP
from equicast.model.layers.positional_embedding import PositionalEmbedder


class EquivariantLayerNorm(nn.Module):
    """LayerNorm for (scalar, vector) pairs. Scalars: standard LN. Vectors: RMS of norms (equivariant)."""

    def __init__(self, hidden_dim: int, eps: float = 1e-5):
        super().__init__()
        self.scalar_norm = nn.LayerNorm(hidden_dim)
        self.eps = eps

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scalar = self.scalar_norm(scalar)
        rms = vector.norm(dim=-1).pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        vector = vector / rms.unsqueeze(-1)
        return scalar, vector


def _multiply_first_dim(w, x):
    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T


class PaiNNMessagePassing(nn.Module):
    """PaiNN-style equivariant message passing with edge direction/length filtering."""

    def __init__(self, hidden_dim: int, aggr: str = "mean"):
        super().__init__()
        self.aggr = aggr
        self.scalar_mlp = MLP(in_dim=hidden_dim, out_dim=3 * hidden_dim, hidden_dim=hidden_dim)
        self.edge_mlp = MLP(in_dim=hidden_dim, out_dim=3 * hidden_dim, hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dirs: torch.Tensor,
        positional_embedding: torch.Tensor,
        src_scalar: torch.Tensor | None = None,
        src_vector: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        num_nodes = scalar.size(0)

        _s = src_scalar if src_scalar is not None else scalar
        _v = src_vector if src_vector is not None else vector
        scalar_j = _s[src]  # [edges, hidden_dim]
        vector_j = _v[src]  # [edges, hidden_dim, 2]
        edge_dirs = edge_dirs.unsqueeze(-2).expand(-1, self.hidden_dim, -1)
        # [edges, hidden_dim, 2]

        scalar_filter = self.edge_mlp(positional_embedding)  # [edges, 3*hidden_dim]
        embedded_scalars = self.scalar_mlp(scalar_j)  # [edges, 3*hidden_dim]
        filtered_scalars = embedded_scalars * scalar_filter  # [edges, 3*hidden_dim]
        scalar_msg, edge_scalers, vector_j_scalers = filtered_scalars.chunk(3, dim=-1)
        scaled_vector_j = _multiply_first_dim(vector_j_scalers, vector_j)
        scaled_edge_dirs = _multiply_first_dim(edge_scalers, edge_dirs)

        vector_msg = scaled_vector_j + scaled_edge_dirs

        d_scalar = scatter(scalar_msg, dst, dim=0, dim_size=num_nodes, reduce=self.aggr)
        d_vector = scatter(vector_msg, dst, dim=0, dim_size=num_nodes, reduce=self.aggr)

        return d_scalar, d_vector


class PaiNNUpdate(nn.Module):
    """PaiNN-style equivariant update block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.U = EquivariantLinear(hidden_dim, hidden_dim)
        self.V = EquivariantLinear(hidden_dim, hidden_dim)
        self.scalar_mlp = MLP(in_dim=2 * hidden_dim, out_dim=3 * hidden_dim, hidden_dim=hidden_dim)

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = self.U(vector)  # [nodes, hidden_dim, 2]
        v = self.V(vector)  # [nodes, hidden_dim, 2]
        v_norm = v.norm(dim=-1)  # [nodes, hidden_dim]

        scalar_in = torch.cat([scalar, v_norm], dim=-1)  # [nodes, 2*hidden_dim]
        add_scalar, u_scalers, norm_filter = self.scalar_mlp(scalar_in).chunk(3, dim=-1)

        d_scalar = v_norm * norm_filter + add_scalar
        d_vector = _multiply_first_dim(u_scalers, u)

        return d_scalar, d_vector


class PaiNNBlock(nn.Module):
    """Combined PaiNN message passing + update block."""

    def __init__(self, hidden_dim: int, aggr: str = "mean"):
        super().__init__()
        self.embed_edge_length = PositionalEmbedder(hidden_dim)
        self.message_passing = PaiNNMessagePassing(hidden_dim, aggr)
        self.norm1 = EquivariantLayerNorm(hidden_dim)
        self.update = PaiNNUpdate(hidden_dim)
        self.norm2 = EquivariantLayerNorm(hidden_dim)

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor, edges) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = edges["edge_index"].long()
        edge_dirs = edges.edge_dirs
        edge_emb = self.embed_edge_length(edges.edge_length)

        d_scalar, d_vector = self.message_passing(scalar, vector, edge_index, edge_dirs, edge_emb)
        scalar, vector = self.norm1(scalar + d_scalar, vector + d_vector)

        d_scalar, d_vector = self.update(scalar, vector)
        return self.norm2(scalar + d_scalar, vector + d_vector)


class PaiNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_vector_dim: int,
        out_vector_dim: int,
        edges: list[tuple[str, str, str]],
        input_nodes: str = "data",
        hidden_dim: int = 64,
        aggr: str = "mean",
        use_node_sharding: bool = False,
    ):
        super().__init__()
        self.edges = [tuple(e) for e in edges]
        self.input_nodes = input_nodes
        self.use_node_sharding = use_node_sharding
        self._sharder = None
        self.embed_scalar_in = MLP(in_dim=in_dim, out_dim=hidden_dim)
        self.embed_scalar_out = MLP(in_dim=hidden_dim, out_dim=out_dim)
        self.embed_vector_in = EquivariantLinear(in_vector_dim, hidden_dim)
        self.embed_vector_out = EquivariantLinear(hidden_dim, out_vector_dim)
        self.blocks = nn.ModuleList([PaiNNBlock(hidden_dim, aggr=aggr) for _ in edges])

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_sharder']
        return state

    def forward(self, graph) -> dict[str, torch.Tensor]:
        scalar = self.embed_scalar_in(graph[self.input_nodes].input_scalar)
        vector = self.embed_vector_in(graph[self.input_nodes].input_vector)
        scalar, vector = self._run_blocks(scalar, vector, graph)
        scalar_out = self.embed_scalar_out(scalar) + graph[self.input_nodes].residual_scalar
        vector_out = self.embed_vector_out(vector) + graph[self.input_nodes].residual_vector
        return {"scalar": scalar_out, "vector": vector_out}

    def _run_blocks(self, scalar, vector, graph):
        sharder = self._get_sharder()
        if sharder:
            for block, edge in zip(self.blocks, self.edges):
                scalar, vector = sharder.run_block(block, scalar, vector, edge, graph[edge])
            scalar, vector = sharder.gather(scalar, vector)
        else:
            for block, edge in zip(self.blocks, self.edges):
                scalar, vector = block(scalar, vector, graph[edge])
        return scalar, vector

    def _get_sharder(self):
        if self.use_node_sharding and getattr(self, '_sharder', None) is None:
            from equicast.utils.sharder import NodeSharder

            self._sharder = NodeSharder.create()
        return getattr(self, '_sharder', None)

"""PaiNN-style equivariant message passing and update layers."""

import warnings

import torch
from torch import nn
from torch_geometric.utils import scatter

from equicast.model.layers.embedding import PositionalEmbedder
from equicast.model.layers.equivariant_conv import EquivariantLinear
from equicast.model.layers.mlp import MLP


class PaiNNMessagePassing(nn.Module):
    """PaiNN-style equivariant message passing with edge direction/length filtering."""

    def __init__(
        self,
        hidden_dim: int,
        aggr: str = "mean",
    ):
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalar: [nodes, hidden_dim] invariant features
            vector: [nodes, hidden_dim, 2] equivariant features
            edge_index: [2, edges] graph connectivity
            edge_dirs: [edges, 2] unit direction vectors
            positional_embedding: [edges, hidden_dim] positional encoding of edge lengths

        Returns:
            (d_scalar, d_vector) residual updates
        """
        src, dst = edge_index
        num_nodes = scalar.size(0)

        scalar_j = scalar[src]  # [edges, hidden_dim]
        vector_j = vector[src]  # [edges, hidden_dim, 2]
        edge_dirs = edge_dirs.unsqueeze(-2).expand(-1, self.hidden_dim, -1)
        # [edges, hidden_dim, 2]

        scalar_filter = self.edge_mlp(positional_embedding)  # [edges, 3*hidden_dim]
        embedded_scalars = self.scalar_mlp(scalar_j)  # [edges, 3*hidden_dim]
        filtered_scalars = embedded_scalars * scalar_filter  # [edges, 3*hidden_dim]
        scalar_msg, edge_scalers, vector_j_scalers = filtered_scalars.chunk(3, dim=-1)
        scaled_vector_j = multiply_first_dim(vector_j_scalers, vector_j)
        scaled_edge_dirs = multiply_first_dim(edge_scalers, edge_dirs)

        vector_msg = scaled_vector_j + scaled_edge_dirs

        # Aggregate to destination nodes
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

    def forward(
        self, scalar: torch.Tensor, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalar: [nodes, hidden_dim]
            vector: [nodes, hidden_dim, 2]

        Returns:
            Updated (scalar, vector)
        """
        u = self.U(vector)  # [nodes, hidden_dim, 2]
        v = self.V(vector)  # [nodes, hidden_dim, 2]
        v_norm = v.norm(dim=-1)  # [nodes, hidden_dim]

        scalar_in = torch.cat([scalar, v_norm], dim=-1)  # [nodes, 2*hidden_dim]
        add_scalar, u_scalers, norm_filter = self.scalar_mlp(scalar_in).chunk(3, dim=-1)

        d_scalar = v_norm * norm_filter
        d_scalar = d_scalar + add_scalar

        d_vector = multiply_first_dim(u_scalers, u)

        return d_scalar, d_vector


class PaiNNBlock(nn.Module):
    """Combined PaiNN message passing + update block."""

    def __init__(
        self,
        hidden_dim: int,
        max_edge_length: float = 10.0,
        aggr: str = "mean",
    ):
        super().__init__()

        self.embed_edge_length = PositionalEmbedder(hidden_dim, max_edge_length)
        self.message_passing = PaiNNMessagePassing(hidden_dim, aggr)
        self.update = PaiNNUpdate(hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edges,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalar: [nodes, hidden_dim]
            vector: [nodes, vector_dim, 2]
            edges: edge store with edge_index, edge_dirs, edge_length

        Returns:
            Updated (scalar, vector) with original dimensions
        """
        edge_index = edges["edge_index"].long()
        edge_dirs = edges.edge_dirs
        edge_emb = self.embed_edge_length(edges.edge_length)

        d_scalar, d_vector = self.message_passing(scalar, vector, edge_index, edge_dirs, edge_emb)

        scalar = scalar + d_scalar
        vector = vector + d_vector

        d_scalar, d_vector = self.update(scalar, vector)

        return scalar + d_scalar, vector + d_vector


def multiply_first_dim(w, x):
    with warnings.catch_warnings(record=True):
        return (w.T * x.T).T

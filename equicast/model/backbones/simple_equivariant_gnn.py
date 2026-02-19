"""Simple single-layer equivariant GNN: linear projections + one EquivariantBlock."""

import torch
from torch import nn

from equicast.data.feature_indices import FeatureIndices
from equicast.model.layers.equivariant_conv import EquivariantBlock


class SimpleEquivariantGNN(nn.Module):
    """Minimal equivariant GNN: linear encoder, single EquivariantBlock, linear decoder."""

    def __init__(
        self,
        feature_indices: FeatureIndices,
        grid_nodes: str = "grid",
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.grid_nodes = grid_nodes

        scalar_in_dim = feature_indices.in_dim
        scalar_out_dim = feature_indices.out_dim
        in_vector_dim = feature_indices.in_vector_dim
        out_vector_dim = feature_indices.out_vector_dim

        # Linear projections to common hidden_dim
        self.scalar_encoder = nn.Linear(scalar_in_dim, hidden_dim)
        self.vector_encoder = nn.Linear(in_vector_dim, hidden_dim, bias=False)

        # Single equivariant block (message passing + update)
        self.processor = EquivariantBlock(
            scalar_dim=hidden_dim,
            vector_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

        # Linear projections back to output dim
        self.scalar_decoder = nn.Linear(hidden_dim, scalar_out_dim)
        self.vector_decoder = nn.Linear(hidden_dim, out_vector_dim, bias=False)

    def forward(self, graph) -> dict[str, torch.Tensor]:
        scalar = graph[self.grid_nodes].input_scalar
        vector = graph[self.grid_nodes].input_vector  # [nodes, vector_dim, 2]
        edge_index = graph[self.grid_nodes, "to", self.grid_nodes]["edge_index"]

        scalar_residual = graph[self.grid_nodes].residual_scalar
        vector_residual = graph[self.grid_nodes].residual_vector

        # Encode
        scalar = self.scalar_encoder(scalar)
        vector = self.vector_encoder(vector.transpose(-1, -2)).transpose(-1, -2)

        # Process
        scalar, vector = self.processor(scalar, vector, edge_index)

        # Decode
        scalar_out = self.scalar_decoder(scalar)
        vector_out = self.vector_decoder(vector.transpose(-1, -2)).transpose(-1, -2)

        return {
            "scalar": scalar_out + scalar_residual,
            "vector": vector_out + vector_residual,
        }

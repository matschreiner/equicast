"""Equivariant GNN backbone for processing scalar and vector features."""

import torch
from torch import nn

from equicast.model.layers.equivariant_conv import EquivariantBlock
from equicast.model.layers.mlp import MLP


class EquivariantGNN(nn.Module):
    """Equivariant GNN that processes both scalar and vector features."""

    def __init__(
        self,
        data_handler,
        grid_nodes: str = "grid",
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.data_handler = data_handler
        self.grid_nodes = grid_nodes

        # Dimensions
        scalar_in_dim = data_handler.in_dim
        scalar_out_dim = data_handler.out_dim
        vector_dim = data_handler.vector_dim

        # Encoder: project to hidden dim
        self.scalar_encoder = MLP(in_dim=scalar_in_dim, out_dim=hidden_dim)
        self.vector_encoder = nn.Linear(vector_dim, hidden_dim, bias=False)

        # Processor: single equivariant block
        self.processor = EquivariantBlock(
            scalar_dim=hidden_dim,
            vector_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )

        # Decoder: project back to output dim
        self.scalar_decoder = MLP(in_dim=hidden_dim, out_dim=scalar_out_dim)
        self.vector_decoder = nn.Linear(hidden_dim, vector_dim, bias=False)

    def forward(self, graph) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            graph: HeteroData with grid node features

        Returns:
            Dict with "scalar" and "vector" outputs
        """
        # Get inputs
        scalar_in = graph[self.grid_nodes].input_scalar
        vector_in = graph[self.grid_nodes].input_vector  # [nodes, vector_dim, 2]
        edge_index = graph[self.grid_nodes, "to", self.grid_nodes]["edge_index"]

        # Get residuals
        scalar_residual = graph[self.grid_nodes].residual_scalar
        vector_residual = graph[self.grid_nodes].residual_vector

        # Encode
        scalar = self.scalar_encoder(scalar_in)
        # Vector encoder: [nodes, vector_dim, 2] -> [nodes, hidden_dim, 2]
        vector = self.vector_encoder(vector_in.transpose(-1, -2)).transpose(-1, -2)

        # Process
        scalar, vector = self.processor(scalar, vector, edge_index)

        # Decode
        scalar_out = self.scalar_decoder(scalar)
        # Vector decoder: [nodes, hidden_dim, 2] -> [nodes, vector_dim, 2]
        vector_out = self.vector_decoder(vector.transpose(-1, -2)).transpose(-1, -2)

        # Add residuals
        scalar_out = scalar_out + scalar_residual
        vector_out = vector_out + vector_residual

        return {"scalar": scalar_out, "vector": vector_out}

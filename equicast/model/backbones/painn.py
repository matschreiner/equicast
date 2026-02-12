"""PaiNN backbone model."""

import torch
from torch import nn

from equicast.model.layers.equivariant_conv import EquivariantLinear
from equicast.model.layers.mlp import MLP
from equicast.model.layers.painn_conv import PaiNNBlock


class PaiNN(nn.Module):
    def __init__(
        self,
        feature_config,
        grid_nodes: str = "grid",
        mesh_nodes: str = "mesh",
        hidden_dim: int = 32,
        aggr: str = "mean",
    ):
        super().__init__()
        self.grid_nodes = grid_nodes
        self.mesh_nodes = mesh_nodes

        scalar_in_dim = len(feature_config.forcing) + len(feature_config.prognostic)
        scalar_out_dim = len(feature_config.prognostic) + len(feature_config.diagnostic)
        vector_dim = len(feature_config.prognostic_vector)

        self.embed_vector_in = EquivariantLinear(vector_dim, hidden_dim)
        self.embed_vector_out = EquivariantLinear(hidden_dim, vector_dim)
        self.embed_scalar_in = MLP(in_dim=scalar_in_dim, out_dim=hidden_dim)
        self.embed_scalar_out = MLP(in_dim=hidden_dim, out_dim=scalar_out_dim)

        self.painnblock1 = PaiNNBlock(hidden_dim, aggr=aggr)
        self.painnblock2 = PaiNNBlock(hidden_dim, aggr=aggr)
        self.painnblock3 = PaiNNBlock(hidden_dim, aggr=aggr)

    def forward(self, graph) -> dict[str, torch.Tensor]:
        scalar_in = graph[self.grid_nodes].input_scalar
        vector_in = graph[self.grid_nodes].input_vector

        embedded_scalar = self.embed_scalar_in(scalar_in)
        embedded_vector = self.embed_vector_in(vector_in)

        edges1 = graph[self.grid_nodes, "to", self.mesh_nodes]
        edges2 = graph[self.mesh_nodes, "to", self.mesh_nodes]
        edges3 = graph[self.mesh_nodes, "to", self.grid_nodes]

        scalar_residual = graph[self.grid_nodes].residual_scalar
        vector_residual = graph[self.grid_nodes].residual_vector

        scalar, vector = self.painnblock1(embedded_scalar, embedded_vector, edges1)
        scalar, vector = self.painnblock2(scalar, vector, edges2)
        scalar, vector = self.painnblock3(scalar, vector, edges3)

        scalar_out = self.embed_scalar_out(scalar) + scalar_residual
        vector_out = self.embed_vector_out(vector) + vector_residual

        return {"scalar": scalar_out, "vector": vector_out}

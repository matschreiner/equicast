"""PaiNN backbone model."""

import torch
from torch import nn

from equicast.data.feature_indices import FeatureIndices
from equicast.model.layers.equivariant_conv import EquivariantLinear
from equicast.model.layers.mlp import MLP
from equicast.model.layers.painn_conv import PaiNNBlock


class PaiNN(nn.Module):
    def __init__(
        self,
        feature_indices: FeatureIndices,
        edges: list[tuple[str, str, str]],
        input_nodes: str = "grid",
        hidden_dim: int = 64,
        aggr: str = "mean",
    ):
        super().__init__()
        self.edges = edges
        self.input_nodes = input_nodes

        scalar_in_dim = feature_indices.in_dim
        scalar_out_dim = feature_indices.out_dim
        in_vector_dim = feature_indices.in_vector_dim
        out_vector_dim = feature_indices.out_vector_dim

        self.embed_scalar_in = MLP(in_dim=scalar_in_dim, out_dim=hidden_dim)
        self.embed_scalar_out = MLP(in_dim=hidden_dim, out_dim=scalar_out_dim)
        self.embed_vector_in = EquivariantLinear(in_vector_dim, hidden_dim)
        self.embed_vector_out = EquivariantLinear(hidden_dim, out_vector_dim)

        self.blocks = nn.ModuleList([PaiNNBlock(hidden_dim, aggr=aggr) for _ in edges])

    def forward(self, graph) -> dict[str, torch.Tensor]:
        scalar = self.embed_scalar_in(graph[self.input_nodes].input_scalar)
        vector = self.embed_vector_in(graph[self.input_nodes].input_vector)

        for block, edge in zip(self.blocks, self.edges):
            scalar, vector = block(scalar, vector, graph[edge])

        scalar_out = self.embed_scalar_out(scalar) + graph[self.input_nodes].residual_scalar
        vector_out = self.embed_vector_out(vector) + graph[self.input_nodes].residual_vector

        return {"scalar": scalar_out, "vector": vector_out}

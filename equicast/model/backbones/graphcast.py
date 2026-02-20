"""GraphCast-style GNN backbone.

Key differences vs the basic GNN:
- Persistent edge state: edges have their own feature vectors, updated each step with residuals
- Receiver node features: edge update sees both src and dst node features
- LayerNorm on nodes and edges after each block
- Sum aggregation
"""

import torch
from anemoi.models.layers.normalization import AutocastLayerNorm
from torch import nn
from torch_geometric.utils import scatter

from equicast.data.feature_index import FeatureIndex
from equicast.model.layers.positional_embedding import PositionalEmbedder
from equicast.model.layers.mlp import MLP


class EdgeEncoder(nn.Module):
    """Encodes edge geometry (length + direction) into hidden_dim."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.pos_embedder = PositionalEmbedder(hidden_dim)
        self.mlp = MLP(in_dim=hidden_dim + 2, out_dim=hidden_dim)

    def forward(self, edge_length: torch.Tensor, edge_dirs: torch.Tensor) -> torch.Tensor:
        pos_emb = self.pos_embedder(edge_length)
        return self.mlp(torch.cat([pos_emb, edge_dirs], dim=-1))


class GraphCastBlock(nn.Module):
    """One message passing step with persistent edge state.

    Edge update: f(src, dst, edge) -> new_edge  (residual + LN)
    Node update: g(node, agg_edges)  -> new_node (residual + LN)
    """

    def __init__(self, hidden_dim: int, aggr: str = "sum"):
        super().__init__()
        self.aggr = aggr
        self.edge_mlp = MLP(in_dim=3 * hidden_dim, out_dim=hidden_dim)
        self.node_mlp = MLP(in_dim=2 * hidden_dim, out_dim=hidden_dim)
        self.edge_norm = AutocastLayerNorm(hidden_dim)
        self.node_norm = AutocastLayerNorm(hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        n_nodes = scalar.size(0)

        # Edge update: condition on both endpoints and current edge state
        edge_in = torch.cat([scalar[src], scalar[dst], edge_feats], dim=-1)
        edge_feats = self.edge_norm(edge_feats + self.edge_mlp(edge_in))

        # Aggregate edges to destination nodes
        agg = scatter(edge_feats, dst, dim=0, dim_size=n_nodes, reduce=self.aggr)

        # Node update
        scalar = self.node_norm(scalar + self.node_mlp(torch.cat([scalar, agg], dim=-1)))

        return scalar, edge_feats


class GraphCast(nn.Module):
    def __init__(
        self,
        feature_index: FeatureIndex,
        edges: list[tuple[str, str, str]],
        input_nodes: str = "grid",
        hidden_dim: int = 64,
        aggr: str = "sum",
    ):
        super().__init__()
        self.edges = edges
        self.input_nodes = input_nodes

        self.embed_in = MLP(in_dim=feature_index.in_dim, out_dim=hidden_dim)
        self.embed_out = MLP(in_dim=hidden_dim, out_dim=feature_index.out_dim)

        # One edge encoder per unique edge type (initialized from geometry)
        unique_edges = list(dict.fromkeys(edges))
        self.edge_encoders = nn.ModuleDict({self._key(e): EdgeEncoder(hidden_dim) for e in unique_edges})

        # One processor block per edge in the list
        self.blocks = nn.ModuleList([GraphCastBlock(hidden_dim, aggr) for _ in edges])

    def _key(self, edge: tuple[str, str, str]) -> str:
        return "__".join(edge)

    def forward(self, graph) -> torch.Tensor:
        scalar = self.embed_in(graph[self.input_nodes].input)

        # Initialize edge features from geometry (once per unique edge type)
        edge_feats = {
            edge: self.edge_encoders[self._key(edge)](
                graph[edge].edge_length,
                graph[edge].edge_dirs,
            )
            for edge in dict.fromkeys(self.edges)
        }

        # Process — edge state is carried forward across blocks of the same type
        for block, edge in zip(self.blocks, self.edges):
            edge_index = graph[edge]["edge_index"].long()
            scalar, edge_feats[edge] = block(scalar, edge_feats[edge], edge_index)

        return self.embed_out(scalar) + graph[self.input_nodes].residual

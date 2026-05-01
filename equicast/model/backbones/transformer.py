"""Graph Transformer backbone with encoder-processor-decoder support.

Supports heterogeneous graphs (e.g. grid→mesh→mesh→grid) by tracking
node features per node type. Non-input node types are initialised to zeros
and built up through message passing.

Architecture per block: pre-norm multi-head graph attention + pre-norm MLP.
Attention: Q[dst], K[src] + E[edge], V[src] aggregated to dst nodes.
"""

import torch
from anemoi.models.layers.normalization import AutocastLayerNorm
from torch import nn
from torch_geometric.utils import scatter, softmax

from equicast.data.feature_index import FeatureIndex
from equicast.model.backbones.graphcast import EdgeEncoder
from equicast.model.graph_encoder import GraphEncoder
from equicast.model.layers.mlp import MLP


class GraphAttention(nn.Module):
    """Multi-head sparse graph attention.

    For each edge (src→dst): attends using Q[dst], K[src]+E[edge], V[src].
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        assert hidden_dim % num_heads == 0, f"{hidden_dim=} must be divisible by {num_heads=}"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.lin_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.lin_out = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x_src: torch.Tensor,      # [N_src, D]
        x_dst: torch.Tensor,      # [N_dst, D]
        edge_attr: torch.Tensor,  # [E, D]
        src_idx: torch.Tensor,    # [E]
        dst_idx: torch.Tensor,    # [E]
    ) -> torch.Tensor:
        N_dst = x_dst.shape[0]
        H, D = self.num_heads, self.head_dim

        q = self.lin_q(x_dst)[dst_idx].view(-1, H, D)   # [E, H, D]
        k = self.lin_k(x_src)[src_idx].view(-1, H, D)   # [E, H, D]
        v = self.lin_v(x_src)[src_idx].view(-1, H, D)   # [E, H, D]
        e = self.lin_e(edge_attr).view(-1, H, D)         # [E, H, D]

        attn = (q * (k + e)).sum(-1) * self.scale        # [E, H]
        attn = softmax(attn, dst_idx, num_nodes=N_dst)   # [E, H] sparse softmax per dst node

        out = scatter(
            attn.unsqueeze(-1) * v,
            dst_idx, dim=0, dim_size=N_dst, reduce="sum",
        )  # [N_dst, H, D]

        return self.lin_out(out.view(N_dst, H * D))


class GraphTransformerBlock(nn.Module):
    """Pre-norm graph transformer block supporting bipartite graphs."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm_src = AutocastLayerNorm(hidden_dim)
        self.norm_dst = AutocastLayerNorm(hidden_dim)
        self.norm_ff = AutocastLayerNorm(hidden_dim)
        self.attention = GraphAttention(hidden_dim, num_heads)
        self.ff = MLP(in_dim=hidden_dim, out_dim=hidden_dim)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,  # [2, E]
    ) -> torch.Tensor:
        src_idx, dst_idx = edge_index

        x_dst = x_dst + self.attention(
            self.norm_src(x_src),
            self.norm_dst(x_dst),
            edge_attr,
            src_idx, dst_idx,
        )
        x_dst = x_dst + self.ff(self.norm_ff(x_dst))
        return x_dst


class GraphTransformer(nn.Module):
    """Graph Transformer backbone with encoder-processor-decoder support.

    Compatible with heterogeneous edge lists like:
        [("data", "to", "hidden"), ("hidden", "to", "hidden"), ("hidden", "to", "data")]

    Non-input node types (e.g. hidden) are initialised to zeros on the first
    forward pass and built up through message passing.
    """

    def __init__(
        self,
        feature_index: FeatureIndex,
        edges: list[tuple[str, str, str]],
        input_nodes: str = "data",
        hidden_dim: int = 128,
        num_heads: int = 4,
        node_encoder: GraphEncoder | None = None,
    ):
        super().__init__()
        self.edges = edges
        self.input_nodes = input_nodes
        self.hidden_dim = hidden_dim

        self.embed_in = MLP(in_dim=feature_index.in_dim, out_dim=hidden_dim)
        self.embed_out = MLP(in_dim=hidden_dim, out_dim=feature_index.out_dim)

        unique_edges = list(dict.fromkeys(edges))
        self.edge_encoders = nn.ModuleDict({
            self._key(e): EdgeEncoder(hidden_dim) for e in unique_edges
        })

        self.blocks = nn.ModuleList([
            GraphTransformerBlock(hidden_dim, num_heads) for _ in edges
        ])
        self.node_encoder = node_encoder

    def _key(self, edge: tuple[str, str, str]) -> str:
        return "__".join(edge)

    def forward(self, graph) -> torch.Tensor:
        node_feats: dict[str, torch.Tensor] = {}
        node_feats[self.input_nodes] = self.embed_in(graph[self.input_nodes].input)

        # Initialise all other node types — from GraphEncoder if provided, else zeros
        latents = self.node_encoder() if self.node_encoder is not None else {}
        for src_type, _, dst_type in self.edges:
            for node_type in (src_type, dst_type):
                if node_type not in node_feats:
                    if node_type in latents:
                        node_feats[node_type] = latents[node_type]
                    else:
                        n = graph[node_type].x.shape[0]
                        node_feats[node_type] = node_feats[self.input_nodes].new_zeros(n, self.hidden_dim)

        edge_feats = {
            edge: self.edge_encoders[self._key(edge)](
                graph[edge].edge_length,
                graph[edge].edge_dirs,
            )
            for edge in dict.fromkeys(self.edges)
        }

        for block, edge in zip(self.blocks, self.edges):
            src_type, _, dst_type = edge
            edge_index = graph[edge]["edge_index"].long()
            node_feats[dst_type] = block(
                node_feats[src_type],
                node_feats[dst_type],
                edge_feats[edge],
                edge_index,
            )

        return self.embed_out(node_feats[self.input_nodes]) + graph[self.input_nodes].residual

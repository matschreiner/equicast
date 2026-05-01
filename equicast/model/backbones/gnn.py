import torch
from torch import nn
from torch_geometric.utils import scatter

from equicast.model.layers.positional_embedding import PositionalEmbedder
from equicast.model.layers.mlp import MLP


class GNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edges: list[tuple[str, str, str]],
        input_nodes: str = "data",
        hidden_dim: int = 64,
        aggr: str = "mean",
        edge_dir_dim: int = 2,
    ):
        super().__init__()
        self.edges = edges
        self.input_nodes = input_nodes

        self.embed_in = MLP(in_dim=in_dim, out_dim=hidden_dim)
        self.embed_out = MLP(in_dim=hidden_dim, out_dim=out_dim)
        self.blocks = nn.ModuleList([Block(hidden_dim, aggr=aggr, edge_dir_dim=edge_dir_dim) for _ in edges])

    def forward(self, graph) -> torch.Tensor:
        scalar = self.embed_in(graph[self.input_nodes].input)

        for block, edge in zip(self.blocks, self.edges):
            scalar = block(scalar, graph[edge])

        return self.embed_out(scalar) + graph[self.input_nodes].residual


class MessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, aggr: str = "mean", edge_dir_dim: int = 2):
        super().__init__()
        self.aggr = aggr
        self.message_mlp = MLP(in_dim=2 * hidden_dim + edge_dir_dim, out_dim=hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        edge_index: torch.Tensor,
        positional_embedding: torch.Tensor,
        edge_dirs: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        scalar_j = scalar[src]
        msg = torch.cat([scalar_j, positional_embedding, edge_dirs], dim=-1)
        scalar_msg = self.message_mlp(msg)

        num_nodes = scalar.size(0)
        return scatter(scalar_msg, dst, dim=0, dim_size=num_nodes, reduce=self.aggr)


class Update(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = MLP(in_dim=hidden_dim, out_dim=hidden_dim)

    def forward(self, scalar: torch.Tensor) -> torch.Tensor:
        return self.mlp(scalar)


class Block(nn.Module):
    def __init__(self, hidden_dim: int, aggr: str = "mean", edge_dir_dim: int = 2):
        super().__init__()
        self.embed_edge_length = PositionalEmbedder(hidden_dim)
        self.message_passing = MessagePassing(hidden_dim, aggr, edge_dir_dim)
        self.update = Update(hidden_dim)

    def forward(self, scalar: torch.Tensor, edges) -> torch.Tensor:
        edge_index = edges["edge_index"].long()
        edge_emb = self.embed_edge_length(edges.edge_length)
        d_scalar = self.message_passing(scalar, edge_index, edge_emb, edges.edge_dirs)
        scalar = scalar + d_scalar
        return scalar + self.update(scalar)

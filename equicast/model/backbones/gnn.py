from typing import Optional

import torch
from torch_geometric.nn.conv import MessagePassing

from equicast.model.layers.mlp import MLP


class GNN(torch.nn.Module):
    def __init__(self, data_handler, grid_nodes="grid", edge_dim: int = 3):
        super().__init__()
        self.data_handler = data_handler

        self.conv = GraphConv(
            in_dim=data_handler.in_dim, out_dim=data_handler.out_dim, edge_dim=edge_dim
        )
        self.grid_nodes = grid_nodes

    def forward(self, graph):
        residual = graph[self.grid_nodes].residual
        x = self.conv(
            graph[self.grid_nodes].input,
            graph[self.grid_nodes, "to", self.grid_nodes],
        )

        return x + residual


class EncProcDec(torch.nn.Module):
    """Encoder-Processor-Decoder using GraphConv layers with grid-mesh structure."""

    def __init__(
        self,
        data_handler,
        grid_nodes: str = "grid",
        mesh_nodes: str = "mesh",
        hidden_dim: int = 256,
        edge_dim: int = 3,
        use_residual: bool = True,
    ):
        super().__init__()
        self.data_handler = data_handler

        self.grid_nodes = grid_nodes
        self.mesh_nodes = mesh_nodes
        self.use_residual = use_residual
        self.encoder = GraphConv(in_dim=data_handler.in_dim, out_dim=hidden_dim, edge_dim=edge_dim)
        self.processor1 = GraphConv(in_dim=hidden_dim, out_dim=hidden_dim, edge_dim=edge_dim)
        self.processor2 = GraphConv(in_dim=hidden_dim, out_dim=hidden_dim, edge_dim=edge_dim)
        self.decoder = GraphConv(in_dim=hidden_dim, out_dim=data_handler.out_dim, edge_dim=edge_dim)

    def forward(self, graph):
        mesh_edges = graph[self.mesh_nodes, "to", self.mesh_nodes]

        x = self.encoder(
            graph[self.grid_nodes].input,
            graph[self.grid_nodes, "to", self.mesh_nodes],
        )
        x = x + self.processor1(x, mesh_edges)
        x = x + self.processor2(x, mesh_edges)
        out = self.decoder(x, graph[self.mesh_nodes, "to", self.grid_nodes])

        if self.use_residual:
            out = out + graph[self.grid_nodes].residual

        return out


class GraphConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 3, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.message_mlp = MLP(
            in_dim=2 * in_dim + edge_dim,
            out_dim=out_dim,
        )
        self.update_mlp = MLP(
            in_dim=out_dim + in_dim,
            out_dim=out_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_storage: dict,
        _: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        edge_index = edge_storage["edge_index"].long()
        edge_attr = torch.cat([edge_storage["edge_dirs"], edge_storage["edge_length"]], dim=-1)

        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):  # type: ignore
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.update_mlp(torch.cat([inputs, x], dim=-1))

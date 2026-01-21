from typing import Optional

import torch
from torch_geometric.nn.conv import MessagePassing

from equicast.model.layers.mlp import MLP


class GNN(torch.nn.Module):
    def __init__(self, feature_config, grid_nodes="grid", edge_dim: int = 3):
        super().__init__()
        in_dim = len(feature_config.forcing) + len(feature_config.prognostic)
        out_dim = len(feature_config.prognostic) + len(feature_config.diagnostic)

        self.conv = GraphConv(in_dim=in_dim, out_dim=out_dim, edge_dim=edge_dim)
        self.grid_nodes = grid_nodes

    def forward(self, graph):
        residual = graph[self.grid_nodes].residual
        x = self.conv(
            graph[self.grid_nodes].input,
            graph[self.grid_nodes, "to", self.grid_nodes],
        )

        return x + residual


class GraphConv(MessagePassing):
    def __init__(
        self, in_dim: int, out_dim: int, edge_dim: int = 3, aggr: str = "mean"
    ):
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
        edge_attr = torch.cat(
            [edge_storage["edge_dirs"], edge_storage["edge_length"]], dim=-1
        )

        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def message(self, x_j, x_i, edge_attr):  # type: ignore
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, inputs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.update_mlp(torch.cat([inputs, x], dim=-1))

from typing import Optional

import pytorch_lightning as pl
import torch
from torch_geometric.nn.conv import MessagePassing

from equicast.model.layers.mlp import MLP


class GNN(torch.nn.Module):
    def __init__(self, feature_config, grid_nodes="grid"):
        super().__init__()
        in_dim = len(feature_config.forcing) + len(feature_config.prognostic)
        out_dim = len(feature_config.prognostic) + len(feature_config.diagnostic)

        self.conv = GraphConv(in_dim=in_dim, out_dim=out_dim)
        self.grid_nodes = grid_nodes

    def forward(self, graph):
        residual = graph[self.grid_nodes].residual
        x = self.conv(
            graph[self.grid_nodes].input,
            graph[self.grid_nodes, "to", self.grid_nodes],
        )

        return x + residual


class GraphConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.mlp = MLP(
            in_dim=2 * in_dim,
            out_dim=out_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_storage: dict,
        _: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:

        edge_index = edge_storage["edge_index"].long()
        out = self.propagate(
            x=x,
            edge_index=edge_index,
        )

        return out

    def message(self, x_j, x_i):  # type: ignore
        return self.mlp(torch.cat([x_i, x_j], dim=-1))

    def update(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

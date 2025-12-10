from typing import Optional

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing


class GraphConvWithEdge(MessagePassing):
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

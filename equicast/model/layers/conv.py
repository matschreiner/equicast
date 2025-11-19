from typing import Optional

import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, OptTensor

from equicast.model.layers.mlp import MLP


class EmbedEdge(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()

        self.mlp = MLP(
            in_dim=3,
            out_dim=out_dim,
            num_layers=2,
        )

    def forward(self, edge_storage) -> torch.Tensor:
        edge_attr_in = torch.cat([edge_storage["edge_dirs"], edge_storage["edge_length"]], dim=-1)
        return self.mlp(edge_attr_in)


class EmbedNode(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=2,
        )

    def forward(self, node_features) -> torch.Tensor:
        return self.mlp(node_features)


class GraphConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.embed_edge = EmbedEdge(out_dim=out_dim)
        self.embed_node = EmbedNode(in_dim=in_dim, out_dim=out_dim)

    def forward(
        self,
        src: torch.Tensor,
        edge_storage: dict,
        size: Optional[tuple[int, int]] = None,
    ) -> torch.Tensor:
        edge_index: torch.Tensor = edge_storage["edge_index"].long()

        edge_attr = self.embed_edge(edge_storage)
        node_attr = self.embed_node(src)

        out = self.propagate(
            edge_index=edge_index,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )

        return out

    def message(self, node_attr_j, edge_attr: torch.Tensor) -> torch.Tensor:
        return node_attr_j * edge_attr

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out

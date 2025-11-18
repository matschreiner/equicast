import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, OptTensor


class GraphConv(MessagePassing):
    def __init__(self, in_src: int, out_channels: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.lin_src = nn.Linear(in_src, out_channels)

    def forward(
        self,
        x: OptPairTensor,
        edge_index: torch.Tensor,
        edge_attr: OptTensor = None,
        size: tuple[int, int] | None = None,
    ):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

    def message(self, x_j, x_i, edge_attr) -> torch.Tensor:
        input = torch.cat([x_j, x_i, edge_attr], dim=-1)
        message = self.lin_src(input)

        return message

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        return aggr_out


def test_bi(batch):
    graph = batch["graph"]
    dth = graph["data", "to", "hidden"]

    edge_attr = dth.edge_dirs

    gc = GraphConv(in_src=7, out_channels=32)
    x_j = batch["condition"]
    x_i = graph["hidden"].x

    out = gc(x=(x_j, x_i), edge_index=dth.edge_index.long(), edge_attr=edge_attr)

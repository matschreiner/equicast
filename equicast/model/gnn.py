from typing import Optional

import pytorch_lightning as pl
import torch

#  from anemoi.models.layers.conv import GraphConv, GraphTransformerConv
from anemoi.models.layers.mlp import MLP
from anemoi.models.layers.normalization import AutocastLayerNorm
from anemoi.utils.config import DotDict
from torch import Tensor, nn
from torch.nn import GELU, Linear
from torch.nn.functional import dropout
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, OptTensor, Size
from torch_geometric.utils import scatter, softmax


class MLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation=GELU(),
        norm=AutocastLayerNorm,
        num_layers=3,
        hidden_dim=None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = out_dim

        layers = []
        for i in range(num_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            output_dim = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(Linear(input_dim, output_dim))
            if i != num_layers - 1:
                layers.append(norm(output_dim))
                layers.append(activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class EmbedEdge(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()

        self.mlp = MLP(
            in_dim=3,
            out_dim=out_dim,
            num_layers=2,
        )

    def forward(self, edge_storage) -> Tensor:
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

    def forward(self, node_features) -> Tensor:
        return self.mlp(node_features)


#  class graphconv(messagepassing):
#      def __init__(self, in_dim, out_dim, aggr="mean"):
#          super().__init__(aggr=aggr)
#          self.embed_edge = embededge(out_dim=out_dim)
#          self.embed_node = embednode(in_dim=in_dim, out_dim=out_dim)
#
#      def forward(
#          self,
#          src: tensor,
#          edge_storage: dict,
#          size: optional[tuple[int, int]] = none,
#      ) -> tensor:
#
#          # forward embed edge
#          #  def forward(self, edge_storage) -> tensor:
#          #      edge_attr_in = torch.cat([edge_storage["edge_dirs"], edge_storage["edge_length"]], dim=-1)
#          #      return self.mlp(edge_attr_in)
#
#      def message(self, ...):
#          # embed nodes and edges
#          # return elementwise multiplication of embeddings


class GraphConv(MessagePassing):
    def __init__(self, in_dim: int, out_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.embed_edge = EmbedEdge(out_dim=out_dim)
        self.embed_node = EmbedNode(in_dim=in_dim, out_dim=out_dim)

    def forward(
        self,
        src: Tensor,
        edge_storage: dict,
        size: Optional[tuple[int, int]] = None,
    ) -> Tensor:
        edge_index: Tensor = edge_storage["edge_index"].long()

        edge_attr = self.embed_edge(edge_storage)
        node_attr = self.embed_node(src)

        out = self.propagate(
            edge_index=edge_index,
            node_attr=node_attr,
            edge_attr=edge_attr,
        )

        return out

    def message(self, node_attr_j, edge_attr: Tensor) -> Tensor:
        return node_attr_j * edge_attr

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out


class GNN(pl.LightningModule):
    def __init__(self, variables, hidden_dim=16):
        super().__init__()
        self.save_hyperparameters()
        out_dim = len(variables.prognostic) + len(variables.diagnostic)
        in_dim = len(variables.forcing) + len(variables.prognostic)

        self.enc = GraphConv(in_dim=in_dim, out_dim=hidden_dim)
        self.proc = GraphConv(in_dim=hidden_dim, out_dim=hidden_dim)
        self.dec = GraphConv(in_dim=hidden_dim, out_dim=out_dim)

    def forward(self, batch):
        graph = batch["graph"]
        x = self.enc(
            batch["condition"],
            graph["data", "to", "hidden"],
            size=(graph["data"], graph["hidden"]),
        )
        x = self.proc(
            x,
            graph["hidden", "to", "hidden"],
            size=(graph["hidden"], graph["hidden"]),
        )
        x = self.dec(
            x,
            graph["hidden", "to", "data"],
            size=(graph["hidden"], graph["data"]),
        )

        return x

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        target = batch["target"]
        loss = ((pred - target) ** 2).mean()
        return loss

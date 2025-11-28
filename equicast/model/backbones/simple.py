import pytorch_lightning as pl
import torch

from equicast.model.layers.mlp import MLP


class Simple(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Simple, self).__init__()
        self.net = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=4,
        )

    def forward(self, graph):
        pred = self.net(graph["data"].cond)
        graph["data"].pred = pred
        return graph

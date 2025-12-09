import pytorch_lightning as pl
import torch

from equicast.model.layers.mlp import MLP


class Simple(torch.nn.Module):
    def __init__(self, feature_config):
        super(Simple, self).__init__()

        in_dim = len(feature_config.forcing) + len(feature_config.prognostic)
        out_dim = len(feature_config.prognostic) + len(feature_config.diagnostic)

        self.net = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            num_layers=4,
        )

    def forward(self, graph):
        __import__("pdb").set_trace()  # TODO delme
        pred = self.net(graph["data"].cond)
        graph["data"].pred = pred
        return graph

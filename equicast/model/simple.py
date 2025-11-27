import pytorch_lightning as pl
import torch

from equicast.model.layers.mlp import MLP


class Simple(torch.nn.Module):
    def __init__(self, features, hidden_dim=64):
        super(Simple, self).__init__()
        in_channels = len(features.forcing) + len(features.prognostic)
        out_channels = len(features.prognostic) + len(features.diagnostic)
        self.net = MLP(
            in_dim=in_channels, out_dim=out_channels, hidden_dim=hidden_dim, num_layers=4
        )

    def forward(self, input):
        data = input["data"].data[:, 0]
        self.net(data)

        out = self.net(input)

        return out

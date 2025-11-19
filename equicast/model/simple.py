import pytorch_lightning as pl
import torch

from equicast.model.layers.mlp import MLP


class Simple(pl.LightningModule):
    def __init__(self, variables, hidden_dim=64):
        super(Simple, self).__init__()
        self.save_hyperparameters()
        in_channels = len(variables.forcing) + len(variables.prognostic)
        out_channels = len(variables.prognostic) + len(variables.diagnostic)
        self.net = MLP(
            in_dim=in_channels, out_dim=out_channels, hidden_dim=hidden_dim, num_layers=4
        )

    def forward(self, batch):
        x = batch["condition"]
        y = self.net(x)
        return y

    def loss(self, pred, target):
        loss = (pred - target) ** 2
        return loss.mean()

    def training_step(self, batch, batch_idx):
        target = batch["target"]

        out = self.forward(batch)
        loss = self.loss(out, target)

        return loss.mean()

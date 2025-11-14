import pytorch_lightning as pl
import torch


class Simple(pl.LightningModule):
    def __init__(self, variables):
        super(Simple, self).__init__()
        self.save_hyperparameters()
        in_channels = len(variables.forcing) + len(variables.prognostic)
        out_channels = len(variables.prognostic) + len(variables.diagnostic)
        self.net = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.net(x)
        return x

    def loss(self, pred, target):
        loss = (pred - target) ** 2
        return loss.mean()

    def training_step(self, batch, batch_idx):
        cond = batch["condition"]
        target = batch["target"]

        out = self.forward(cond)
        loss = self.loss(out, target)

        return loss.mean()

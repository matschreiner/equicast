import pytorch_lightning as pl
import torch


class Simple(pl.LightningModule):
    def __init__(self, variables):
        super(Simple, self).__init__()
        self.save_hyperparameters()
        in_channels = len(variables.forcing) + len(variables.prognostic)
        out_channels = len(variables.prognostic) + len(variables.diagnostic)
        self.net = torch.nn.Linear(in_channels, out_channels)

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

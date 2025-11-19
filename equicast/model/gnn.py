import pytorch_lightning as pl
from torch import nn

from equicast.model.layers.conv import GraphConv


class GNN(pl.LightningModule):
    def __init__(self, variables, hidden_dim=16):
        super().__init__()
        self.save_hyperparameters()
        out_dim = len(variables.prognostic) + len(variables.diagnostic)
        in_dim = len(variables.forcing) + len(variables.prognostic)

        self.enc = GraphConv(in_dim=in_dim, out_dim=out_dim)
        #  self.dec = GraphConv(in_dim=hidden_dim, out_dim=out_dim)

    def forward(self, batch):
        graph = batch["graph"]

        x = self.enc(
            batch["condition"],
            graph["data", "to", "data"],
            size=(graph["data"], graph["data"]),
        )

        return x

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        target = batch["target"]
        loss = ((pred - target) ** 2).mean()
        return loss

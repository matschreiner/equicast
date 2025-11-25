import pytorch_lightning as pl

from equicast.model.layers.conv import GraphConv


class GNN(pl.LightningModule):
    def __init__(self, features, preprocess=None):
        super().__init__()
        self.save_hyperparameters()
        out_dim = len(features.prognostic) + len(features.diagnostic)
        in_dim = len(features.forcing) + len(features.prognostic)
        self.preprocess = preprocess
        self.conv = GraphConv(in_dim=in_dim, out_dim=out_dim)

    def forward(self, batch):
        if self.preprocess is not None:
            batch = self.preprocess(batch)

        graph = batch["graph"]

        x = self.conv(
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

import pytorch_lightning as pl

from equicast.model.layers.conv import GraphConv


class GNN(pl.LightningModule):
    def __init__(self, feature_config):
        super().__init__()
        self.save_hyperparameters()
        in_dim = len(feature_config.forcing) + len(feature_config.prognostic)
        out_dim = len(feature_config.prognostic) + len(feature_config.diagnostic)

        self.conv = GraphConv(in_dim=in_dim, out_dim=out_dim)

    def forward(self, graph):
        x = self.conv(
            graph["cond"],
            graph["data", "to", "data"],
            size=(graph["data"], graph["data"]),
        )

        return x

    def training_step(self, batch, _):
        pred = self.forward(batch)
        target = batch["target"]
        loss = ((pred - target) ** 2).mean()
        return loss

import pytorch_lightning as pl

from equicast.data.feature_router import FeatureRouter


class Model(pl.LightningModule):
    def __init__(
        self,
        backbone,
        statistics,
        name_to_index,
        features=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.features = features
        self.name_to_index = name_to_index
        self.statistics = statistics
        self.feature_router = FeatureRouter(features, name_to_index)

    def forward(self, batch):
        cond = batch["cond"]
        #  target = batch["data"].target
        self.prepare_input(cond)

        graph = self.feature_router(batch["data"])

        __import__("pdb").set_trace()  # TODO delme

        self.backbone(graph)

        return

    def prepare_input(self, graph):
        graph.scalar_features = graph.cond

    def training_step(self, batch, batch_idx):
        pred = self.forward(batch)
        target = batch["target"]
        loss = ((pred - target) ** 2).mean()
        return loss

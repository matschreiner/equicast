import warnings
from contextlib import contextmanager

import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(
        self,
        backbone,
        scaler,
        feature_router,
    ):
        super().__init__()
        with ignore_backbone_warning():
            self.save_hyperparameters()

        self.backbone = backbone
        self.scaler = scaler
        self.feature_router = feature_router

    def forward(self, graph):
        graph = self.prepare_input(graph)
        graph = self.backbone(graph)
        return graph

    def forecast(self, graph):
        graph = self.prepare_input(graph)
        pred = self.forward(graph)
        return pred

    def prepare_input(self, graph):
        graph = self.scaler(graph)
        graph = self.feature_router(graph)
        return graph

    def training_step(self, graph, batch_idx):
        graph = self.forward(graph)

        pred = graph["data"].pred
        target = graph["data"].target

        loss = ((pred - target) ** 2).mean()
        return loss


@contextmanager
def ignore_backbone_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Attribute 'backbone' is an instance of `nn\.Module`",
        )
        yield

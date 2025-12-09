import warnings
from contextlib import contextmanager

import pytorch_lightning as pl
import torch


class Model(pl.LightningModule):
    def __init__(
        self,
        backbone,
        optimizer_factory=None,
        scheduler_factory=None,
    ):
        super().__init__()
        with ignore_backbone_warning():
            self.save_hyperparameters()

        self.backbone = backbone
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

    def forward(self, batch):
        batch = self.backbone(batch)
        return batch

    def training_step(self, batch, batch_idx):
        batch = self.forward(batch)
        target = batch["data"].target
        pred = batch["data"].pred

        loss = ((pred - target) ** 2).mean()
        return loss

    def configure_optimizers(self):
        if self.optimizer_factory is None:
            self.optimizer_factory = lambda params: torch.optim.Adam(
                params, lr=1e-3
            )
        optimizer = self.optimizer_factory(self.parameters())
        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer)
            return [optimizer], [scheduler]
        return optimizer


@contextmanager
def ignore_backbone_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Attribute 'backbone' is an instance of `nn\.Module`",
        )
        yield

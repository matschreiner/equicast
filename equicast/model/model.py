import warnings
from contextlib import contextmanager
from typing import Callable

import pytorch_lightning as pl
import torch


class Model(pl.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        optimizer_factory: Callable | None = None,
        scheduler_factory: Callable | None = None,
    ):
        super().__init__()
        with ignore_backbone_warning():
            self.save_hyperparameters()

        self.backbone = backbone
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

    def forward(self, graph):
        graph = self.backbone(graph)
        return graph

    def training_step(self, graph, _):
        pred = self.forward(graph)
        target = graph["grid"].target

        loss = ((pred - target) ** 2).mean()

        self.log_dict(
            {"loss": loss},
            logger=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=graph.num_graphs,
        )

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


def get_lr(self):
    opt = self.trainer.optimizers[0]
    lr = opt.param_groups[0]["lr"]
    return lr

import warnings
from contextlib import contextmanager
from typing import Callable

import pytorch_lightning as pl
import torch

from equicast.data.data_handler import DataHandler


class Model(pl.LightningModule):
    """
    Model that handles preprocessing (scaling, feature routing) internally.

    This makes the model self-contained and ensures consistent preprocessing
    between training and inference.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        data_handler: DataHandler,
        optimizer_factory: Callable | None = None,
        scheduler_factory: Callable | None = None,
    ):
        super().__init__()
        with ignore_backbone_warning():
            self.save_hyperparameters()

        self.backbone = backbone
        self.data_handler = data_handler
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory

    def forward(self, graph):
        """
        Forward pass with preprocessing.

        Args:
            graph: Graph with raw input data in graph["grid"].input_state

        Returns:
            Predictions in scaled space
        """

        cond = self.data_handler.prepare_model_input(graph["grid"].input_state)
        graph["grid"].cond = cond
        pred = self.backbone(graph)

        return pred

    def training_step(self, graph, _):
        pred = self.forward(graph)
        target = self.data_handler.prepare_model_target(
            graph["grid"].target_state
        )

        loss = ((pred - target) ** 2).mean()
        self.log_loss(loss, graph.num_graphs)

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

    def log_loss(self, loss, batch_size):
        lr = get_lr(self)
        log_dict = {"loss": loss, "lr": lr}

        self.log_dict(
            log_dict,
            logger=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )


@contextmanager
def ignore_backbone_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Attribute '(backbone|data_handler)' is an instance of `nn\.Module`",
        )
        yield


def get_lr(self):
    opt = self.trainer.optimizers[0]
    lr = opt.param_groups[0]["lr"]
    return lr

import warnings
from contextlib import contextmanager
from typing import Callable

import pytorch_lightning as pl
import torch


class Model(pl.LightningModule):
    """
    Model that handles preprocessing (scaling, feature routing) internally.

    This makes the model self-contained and ensures consistent preprocessing
    between training and inference.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        data_handler,
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
        # Preprocess: scale and route input
        input_scaled = self.data_handler.scaler(graph["grid"].input_state)
        cond = input_scaled[:, self.data_handler.in_idxs]

        # Store processed input in graph for backbone
        graph["grid"].cond = cond

        # Backbone forward pass
        pred = self.backbone(graph)

        return pred

    def training_step(self, graph, _):
        # Get prediction from forward pass
        pred = self.forward(graph)

        # Process target separately (only needed for training)
        target_scaled = self.data_handler.scaler(graph["grid"].target_state)
        target = target_scaled[:, self.data_handler.out_idxs]

        # Compute loss
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

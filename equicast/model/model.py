import warnings
from contextlib import contextmanager
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch

from equicast.data.data_handler import BaseDataHandler
from equicast.metrics import BaseMetricsTracker


def default_optimizer_factory(params):
    return torch.optim.Adam(params, lr=1e-3)


class Model(pl.LightningModule):
    """
    Model that handles preprocessing (scaling, feature routing) internally.

    This makes the model self-contained and ensures consistent preprocessing
    between training and inference.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        data_handler: BaseDataHandler,
        optimizer_factory: Callable = default_optimizer_factory,
        scheduler_factory: Callable | None = None,
        metrics_tracker: BaseMetricsTracker | None = None,
    ):
        super().__init__()
        with ignore_backbone_warning():
            self.save_hyperparameters()

        self.backbone = backbone
        self.data_handler = data_handler
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.metrics_tracker = metrics_tracker

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data):
        data = self.data_handler.prepare_input(data)
        pred = self.backbone.forward(data)
        pred = self.data_handler.inverse_normalize_output_features(pred)
        return pred

    def step_forward(self, condition, next_graph):
        """Single autoregressive step.

        Args:
            condition: Data to make prediction from
            next_graph: Graph to update with prediction (provides forcing)

        Returns:
            tuple: (updated next_graph, prediction tensor)
        """
        pred = self.forward(condition)
        next_graph = self.data_handler.update_next_with_prediction(next_graph, pred)
        return next_graph, pred

    def training_step(self, graph, _):
        graph = self.data_handler.prepare_input(graph)
        pred = self.backbone.forward(graph)

        target = self.data_handler.get_target(graph)
        loss = self.loss(pred, target)

        self.log_loss(loss, graph.num_graphs)
        self.log_metrics(pred, target, graph.num_graphs)

        return loss

    def loss(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())

        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer)
            scheduler_config = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        return optimizer

    def log_loss(self, loss, batch_size):
        """Log loss to progress bar and logger."""
        self.log(
            "loss",
            loss,
            logger=True,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )

    def log_metrics(self, pred, target, batch_size):
        """Log all metrics (lr, log_step, model metrics) to logger only."""
        lr = get_lr(self)
        ln_step = np.log(self.global_step + 1)
        log_dict = {"lr": lr, "log_step": ln_step}

        if self.metrics_tracker is not None:
            metrics = self.metrics_tracker.compute_metrics(pred, target)
            log_dict.update(metrics)

        self.log_dict(
            log_dict,
            logger=True,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
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


def get_lr(model):
    opt = model.trainer.optimizers[0]
    lr = opt.param_groups[0]["lr"]
    return lr

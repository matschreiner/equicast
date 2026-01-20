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

    @property
    def nodes(self):
        return self.data_handler.nodes

    def forward(self, input):
        input = self.data_handler.prepare_input(input)
        backbone_out = self.backbone.forward(input)
        output = self.data_handler.update_output(input, backbone_out)
        return output

    def step_forward(self, input, next):
        """Single autoregressive step.

        Args:
            input: Graph to make prediction from
            next: Graph to update with prediction (provides forcing)

        Returns:
            tuple: (updated next graph, prediction graph)
        """
        input = input.clone()
        next = next.clone()

        pred = self.forward(input)
        next = self.data_handler.update_next_with_prediction(next, pred)
        return next, pred

    def training_step(self, batch, _):
        input = batch["input"]
        input = self.data_handler.prepare_input(input)

        target = batch["target"]
        backbone_target = self.data_handler.prepare_backbone_target(target)

        backbone_out = self.backbone.forward(input)

        loss = self.loss(backbone_out, backbone_target)
        self.log_loss(loss, input.num_graphs)
        self.log_metrics(backbone_out, backbone_target, input.num_graphs)

        return loss

    def loss(self, backbone_out, backbone_target):
        return torch.nn.functional.mse_loss(backbone_out, backbone_target)

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

    def log_metrics(self, backbone_out, backbone_target, batch_size):
        """Log all metrics (lr, log_step, model metrics) to logger only."""
        lr = get_lr(self)
        ln_step = np.log(self.global_step + 1)
        log_dict = {"lr": lr, "log_step": ln_step}

        if self.metrics_tracker is not None:
            metrics = self.metrics_tracker.compute_metrics(
                backbone_out, backbone_target
            )
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

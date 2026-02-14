import time
import warnings
from contextlib import contextmanager
from typing import Callable

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig

from equicast.metrics import BaseMetricsTracker


def default_optimizer_factory(params):
    return torch.optim.Adam(params, lr=1e-3)


def default_loss_fn(backbone_out, backbone_target):
    return torch.nn.functional.mse_loss(backbone_out, backbone_target)


def equivariant_loss_fn(backbone_out, backbone_target):
    """Loss function for equivariant models with scalar and vector outputs."""
    scalar_loss = torch.nn.functional.mse_loss(backbone_out["scalar"], backbone_target["scalar"])
    vector_loss = torch.nn.functional.mse_loss(backbone_out["vector"], backbone_target["vector"])
    return scalar_loss + vector_loss


class Model(pl.LightningModule):
    """
    Model that handles preprocessing (scaling, feature routing) internally.

    This makes the model self-contained and ensures consistent preprocessing
    between training and inference.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        optimizer_factory: Callable = default_optimizer_factory,
        scheduler_factory: Callable | None = None,
        metrics_tracker: BaseMetricsTracker | None = None,
        loss_fn: Callable = default_loss_fn,
        compile_backbone: bool = True,
    ):
        super().__init__()
        with ignore_backbone_warning():
            self.save_hyperparameters()

        self.backbone = torch.compile(backbone) if compile_backbone else backbone
        self.data_handler = backbone.data_handler
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.metrics_tracker = metrics_tracker
        self.loss_fn = loss_fn

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def nodes(self):
        return self.data_handler.nodes

    def forward(self, input):
        input = input.clone()
        input = self.data_handler.prepare_backbone_input(input)
        backbone_out = self.backbone.forward(input)
        output = self.data_handler.update_state_with_backbone_output(input, backbone_out)
        return output

    def step_forward(self, input, next):
        input = input.clone()
        next = next.clone()

        pred = self.forward(input)
        next = self.data_handler.update_state_with_prediction(next, pred)
        return next, pred

    def training_step(self, batch, _):
        self._step_start = time.time()
        if hasattr(self, "_last_step_end"):
            wait_time = self._step_start - self._last_step_end
            self.log("train/wait_time", wait_time, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if hasattr(self, "_last_optimization_step_time"):
            self.log("optimization_step_time", self._last_optimization_step_time, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        input = batch["input"]
        input = self.data_handler.prepare_backbone_input(input)  # type: ignore

        target = batch["target"]
        target = self.data_handler.prepare_backbone_target(target)

        backbone_out = self.backbone.forward(input)

        loss = self.loss(backbone_out, target)
        self.log_loss(loss, input.num_graphs)
        if self.metrics_tracker is not None:
            self.log_metrics(backbone_out, target, input.num_graphs)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self._last_step_end = time.time()
        self._last_optimization_step_time = self._last_step_end - self._step_start

    def loss(self, backbone_out, backbone_target):
        return self.loss_fn(backbone_out, backbone_target)

    def configure_optimizers(self):
        optimizer = self.optimizer_factory(self.parameters())

        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory(optimizer)
            scheduler_config: LRSchedulerConfigType = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
            return OptimizerLRSchedulerConfig(optimizer=optimizer, lr_scheduler=scheduler_config)
        return optimizer

    def log_loss(self, loss, batch_size):
        """Log loss to progress bar and logger."""
        self.log(
            "train/loss",
            loss,
            logger=True,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )

    def log_metrics(self, backbone_out, backbone_target, batch_size):
        """Log all metrics (lr, log_step, model metrics) to logger only."""
        if not self._should_log_metrics():
            return

        lr = get_lr(self)
        ln_step = np.log(self.global_step + 1)
        log_dict = {"train/lr": lr, "train/log_step": ln_step}

        if self.metrics_tracker is not None:
            metrics = self.metrics_tracker.compute_metrics(backbone_out, backbone_target)
            log_dict.update(metrics)

        self.log_dict(
            log_dict,
            logger=True,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )

    def _should_log_metrics(self) -> bool:
        step = self.global_step
        if step < 100:
            return True
        if step < 1000:
            return step % 10 == 0
        return step % 50 == 0


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

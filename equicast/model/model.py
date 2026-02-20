import warnings
from contextlib import contextmanager
from typing import Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig

from equicast.data.data_handler import BaseDataHandler
from equicast.metrics import BaseMetricsTracker
from equicast.model.losses import EquivariantMSELoss, MSELoss


def default_optimizer_factory(params):
    return torch.optim.Adam(params, lr=1e-3)


# Alias for backwards compatibility with old checkpoints
equivariant_loss_fn = EquivariantMSELoss


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
        loss_fn: torch.nn.Module = MSELoss(),
        compile_backbone: bool = True,
    ):
        super().__init__()
        with ignore_backbone_warning():
            self.save_hyperparameters(ignore=["loss_fn"])

        self.backbone = torch.compile(backbone) if compile_backbone else backbone
        self.data_handler = data_handler
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.metrics_tracker = metrics_tracker
        self.loss_fn = loss_fn

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_):
        input_ = input_.clone()
        input_ = self.data_handler.prepare_backbone_input(input_)
        backbone_out = self.backbone(input_)
        output = self.data_handler.update_state_with_backbone_output(input_, backbone_out)
        return output

    def step_forward(self, input_, next_):
        input_ = input_.clone()
        pred = input_.clone()
        next_ = next_.clone()

        input_ = self.data_handler.prepare_backbone_input(input_)
        backbone_out = self.backbone(input_)

        pred = self.data_handler.update_state_with_backbone_output(pred, backbone_out)
        next_ = self.data_handler.update_state_with_backbone_output(next_, backbone_out)
        return next_, pred

    def validation_step(self, batch, _):
        input_ = batch["input"]
        input_ = self.data_handler.prepare_backbone_input(input_)

        target = batch["target"]
        target = self.data_handler.prepare_backbone_target(target)

        backbone_out = self.backbone(input_)
        loss = self.loss(backbone_out, target)

        self.log(
            "val/loss", loss, logger=True, prog_bar=False, on_step=False, on_epoch=True, batch_size=input_.num_graphs
        )
        return loss

    def training_step(self, batch, _):
        input_ = batch["input"]
        input_ = self.data_handler.prepare_backbone_input(input_)

        target = batch["target"]
        target = self.data_handler.prepare_backbone_target(target)

        backbone_out = self.backbone(input_)
        loss = self.loss(backbone_out, target)

        if self._should_log_metrics():
            self.log_lr()
            self.log_loss(loss, input_.num_graphs)

            if self.metrics_tracker is not None:
                pred = self.data_handler.update_state_with_backbone_output(input_, backbone_out)

                metrics = self.metrics_tracker.compute_metrics(pred, target)
                self.log_dict(
                    metrics,
                )

        return loss

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
        self.log("train/loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size)

    def log_lr(self):
        lr = get_lr(self)
        self.log("train/lr", lr, logger=True, prog_bar=True, on_step=True, on_epoch=False)

    def log_metrics(self, backbone_out, backbone_target, batch_size):
        if self.metrics_tracker is None:
            return

        metrics = self.metrics_tracker.compute_metrics(backbone_out, backbone_target)
        self.log_dict(metrics, logger=True, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size)

    def _should_log_metrics(self) -> bool:
        step = self.global_step
        interval = min(100, max(1, step // 100))
        return step % interval == 0


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

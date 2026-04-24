from abc import abstractmethod
from contextlib import contextmanager
from typing import Callable

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig

from equicast.data.data_handler import BaseDataHandler
from equicast.metrics import BaseMetricsTracker


def default_optimizer_factory(params):
    return torch.optim.Adam(params, lr=1e-3)


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        data_handler: BaseDataHandler,
        optimizer_factory: Callable = default_optimizer_factory,
        scheduler_factory: Callable | None = None,
        metrics_tracker: BaseMetricsTracker | None = None,
    ):
        super().__init__()
        self.data_handler = data_handler
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.metrics_tracker = metrics_tracker

    @abstractmethod
    def training_step(self, batch, batch_idx):
        ...

    @abstractmethod
    def predict(self, input_):
        """Accept a physical input graph, return a physical output graph."""
        ...

    def step_forward(self, physical_input, physical_next):
        """Single autoregressive step: predict from physical_input, write output into physical_next."""
        pred = self.predict(physical_input)
        physical_next = physical_next.clone()
        nodes = self.data_handler.nodes
        physical_out = self.data_handler.get_output_scalars(pred[nodes].data)
        self.data_handler.set_output_scalars(physical_next[nodes].data, physical_out)
        return physical_next, pred

    @property
    def device(self):
        return next(self.parameters()).device

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

    def on_before_optimizer_step(self, optimizer):
        if self._should_log_metrics():
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=float("inf"))
            self.log("train/grad_norm", grad_norm, logger=True, prog_bar=False, on_step=True, on_epoch=False)

    def log_loss(self, loss, batch_size):
        self.log("train/loss", loss, logger=True, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size)

    def log_lr(self):
        opt = self.trainer.optimizers[0]
        lr = opt.param_groups[0]["lr"]
        self.log("train/lr", lr, logger=True, prog_bar=True, on_step=True, on_epoch=False)

    def log_metrics(self, backbone_out, backbone_target):
        if self.metrics_tracker is None:
            return
        metrics = self.metrics_tracker.compute_metrics(backbone_out, backbone_target)
        self.log_dict(metrics, logger=True, prog_bar=True, on_step=True, on_epoch=False)

    def _should_log_metrics(self) -> bool:
        step = self.global_step
        interval = min(100, max(1, step // 100))
        return step % interval == 0


@contextmanager
def ignore_module_warning(*attr_names):
    pattern = "|".join(attr_names)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=rf"Attribute '({pattern})' is an instance of `nn\.Module`",
        )
        yield

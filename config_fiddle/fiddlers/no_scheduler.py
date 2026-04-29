import fiddle as fdl
import pytorch_lightning as pl


class StripSchedulerCheckpoint(pl.Callback):
    """Strip lr_schedulers from checkpoint when resuming without a scheduler."""

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["lr_schedulers"] = []


def fiddler(cfg: fdl.Config) -> None:
    cfg.model.scheduler_factory = None
    cfg.trainer.callbacks.append(fdl.Config(StripSchedulerCheckpoint))

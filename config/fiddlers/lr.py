import fiddle as fdl
import pytorch_lightning as pl


class SetLRCallback(pl.Callback):
    def __init__(self, lr: float):
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        for pg in trainer.optimizers[0].param_groups:
            pg["lr"] = self.lr


def fiddler(cfg: fdl.Config, lr: str = "1e-6") -> None:
    lr_float = float(lr)
    cfg.trainer.callbacks.append(fdl.Config(SetLRCallback, lr=lr_float))

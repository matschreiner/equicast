from types import MethodType

import pytorch_lightning as pl


class Trainer(pl.Trainer):
    def __init__(self, optimizer=None, scheduler=None, *args, **kwargs):
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.scheduler = None

        super().__init__(*args, **kwargs)

    def fit(
        self,
        model,
        *args,
        **kwargs,
    ):
        model.configure_optimizers = MethodType(
            self.configure_optimizers_callback(), model
        )
        model.on_train_batch_end = MethodType(self.on_train_batch_end, model)

        super().fit(model, *args, **kwargs)

    def configure_optimizers_callback(self):
        def configure_optimizers(*args, **kwargs):
            if self.scheduler is not None:
                return [self.optimizer], [self.scheduler]
            return self.optimizer

        return configure_optimizers

    @staticmethod
    def on_train_batch_end(pl_module, loss, batch, dataloader_idx):
        pl_module.log("loss", loss["loss"], prog_bar=True, on_step=True, on_epoch=True)
        log_lr(pl_module)


def log_lr(pl_module):
    opt = pl_module.trainer.optimizers[0]
    lr = opt.param_groups[0]["lr"]
    pl_module.log(
        "lr",
        lr,
        on_step=True,
        on_epoch=False,  # usually makes sense only per step
    )

from types import MethodType

import pytorch_lightning as pl


class Trainer(pl.Trainer):
    def __init__(self, optimizer=None, scheduler=None, *args, **kwargs):
        self.scheduler = scheduler
        self.optimizer = optimizer

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
        pl_module.log(
            "train/loss", loss["loss"], prog_bar=True, on_step=True, on_epoch=True
        )

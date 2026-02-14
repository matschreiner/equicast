import pytorch_lightning as pl
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class EMA(pl.Callback):
    """Exponential Moving Average using torch's AveragedModel."""

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_model: AveragedModel | None = None
        self._swapped = False

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_model is None:
            self.ema_model = AveragedModel(pl_module, multi_avg_fn=get_ema_multi_avg_fn(self.decay))

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args) -> None:
        self.ema_model.update_parameters(pl_module)

    def _swap_params(self, pl_module: pl.LightningModule) -> None:
        for ema_p, model_p in zip(self.ema_model.module.parameters(), pl_module.parameters()):
            tmp = model_p.data.clone()
            model_p.data.copy_(ema_p.data)
            ema_p.data.copy_(tmp)
        self._swapped = not self._swapped

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_model is not None and not self._swapped:
            self._swap_params(pl_module)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_model is not None and self._swapped:
            self._swap_params(pl_module)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_model is not None and not self._swapped:
            self._swap_params(pl_module)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_model is not None and self._swapped:
            self._swap_params(pl_module)

    def on_predict_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_model is not None and not self._swapped:
            self._swap_params(pl_module)

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.ema_model is not None and self._swapped:
            self._swap_params(pl_module)

    def on_save_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        if self.ema_model is not None:
            if self._swapped:
                # Params are swapped: pl_module has EMA weights, ema_model has training weights.
                # Undo the swap in the checkpoint so the saved state is always canonical
                # (model_state_dict = training weights, ema_state_dict = EMA weights).
                checkpoint["ema_state_dict"] = {k: v.clone() for k, v in pl_module.state_dict().items()}
                checkpoint["state_dict"] = self.ema_model.module.state_dict()
            else:
                checkpoint["ema_state_dict"] = self.ema_model.module.state_dict()

    def on_load_checkpoint(self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: dict) -> None:
        if "ema_state_dict" in checkpoint:
            if self.ema_model is None:
                self.ema_model = AveragedModel(pl_module, multi_avg_fn=get_ema_multi_avg_fn(self.decay))
            self.ema_model.module.load_state_dict(checkpoint["ema_state_dict"])

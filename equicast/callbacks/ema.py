import copy

import torch
from pytorch_lightning import Callback


class EMA(Callback):
    """Exponential Moving Average of model weights.

    Maintains shadow weights updated each step as:
        ema_weight = decay * ema_weight + (1 - decay) * model_weight

    Swaps EMA weights into the model before checkpointing/validation,
    then swaps back for continued training.
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}

    def on_train_start(self, trainer, pl_module):
        self.shadow = {name: p.clone().detach() for name, p in pl_module.named_parameters()}

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, p in pl_module.named_parameters():
            self.shadow[name].lerp_(p.data, 1 - self.decay)

    def _swap(self, pl_module):
        for name, p in pl_module.named_parameters():
            tmp = p.data.clone()
            p.data.copy_(self.shadow[name])
            self.shadow[name].copy_(tmp)

    def on_validation_start(self, trainer, pl_module):
        self._swap(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self._swap(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self._swap(pl_module)
        checkpoint["state_dict"] = copy.deepcopy(pl_module.state_dict())
        self._swap(pl_module)
        checkpoint["ema_shadow"] = self.shadow

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if "ema_shadow" in checkpoint:
            self.shadow = checkpoint.pop("ema_shadow")

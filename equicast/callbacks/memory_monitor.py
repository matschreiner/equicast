import torch
from pytorch_lightning import Callback


class MemoryMonitor(Callback):
    def on_train_start(self, trainer, pl_module):
        torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            peak_gb = torch.cuda.max_memory_allocated() / 1e9
            pl_module.log("system/peak_memory_gb", peak_gb, logger=True, prog_bar=False, on_step=True, on_epoch=False)

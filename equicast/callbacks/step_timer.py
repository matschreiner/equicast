import time

from pytorch_lightning import Callback


class StepTimer(Callback):
    """Logs wall-clock wait_time and step_time per training step."""

    def __init__(self, start_step: int = 10):
        super().__init__()
        self.start_step = start_step
        self._step_start = None
        self._last_step_end = None
        self._last_opt_time = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._last_step_end is not None and trainer.global_step >= self.start_step:
            wait_time = time.time() - self._last_step_end
            it_per_s = 1.0 / (wait_time + self._last_opt_time)
            pl_module.log("train/wait_time", wait_time, prog_bar=True, logger=True, on_step=True, on_epoch=False)
            pl_module.log(
                "train/opt_step_time", self._last_opt_time, prog_bar=True, logger=True, on_step=True, on_epoch=False
            )
            pl_module.log("performance/it_per_s", it_per_s, prog_bar=False, logger=True, on_step=True, on_epoch=False)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        self._last_opt_time = now - self._step_start
        self._last_step_end = now

    def mark_step_start(self):
        """Called from training_step to mark the start of the optimization step."""
        self._step_start = time.time()

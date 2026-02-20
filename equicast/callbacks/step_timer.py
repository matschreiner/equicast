import time

from pytorch_lightning import Callback


class StepTimer(Callback):
    """Logs wall-clock wait_time and batch_time per training step.

    Metrics are logged with a one-step lag: timings recorded during step N
    are logged at the start of step N+1, once total_step_time is known.
    The first `start_step` steps are skipped to avoid CUDA warmup noise.
    """

    def __init__(self, start_step: int = 10):
        super().__init__()
        self.start_step = start_step
        self._step_start = None
        self._last_step_end = None
        self._last_batch_time = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):  # type: ignore
        self._step_start = time.time()
        if self._last_step_end is None or trainer.global_step < self.start_step:
            return
        wait_time = self._step_start - self._last_step_end
        total_time = wait_time + self._last_batch_time
        if total_time == 0:
            return
        pl_module.log_dict(
            {
                "performance/wait_time": wait_time,
                "performance/batch_time": self._last_batch_time,
                "performance/total_step_time": total_time,
                "performance/it_per_s": 1.0 / total_time,
            },
            prog_bar=False,
            logger=True,
            on_step=True,
            on_epoch=False,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        self._last_batch_time = now - self._step_start
        self._last_step_end = now

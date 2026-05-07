import os
import time

from pytorch_lightning import Callback


class CheckpointSaver:
    def __init__(self, dirpath: str, logger=None):
        self.dirpath = dirpath
        self.logger = logger
        os.makedirs(dirpath, exist_ok=True)

    def save(self, trainer, filename: str):
        filepath = os.path.join(self.dirpath, filename)
        trainer.save_checkpoint(filepath)
        if trainer.is_global_zero and self.logger and hasattr(self.logger, "after_save_checkpoint"):
            self.logger.after_save_checkpoint(filepath)

    @classmethod
    def from_trainer(cls, trainer) -> "CheckpointSaver":
        logger = trainer.logger
        run_id = getattr(logger, "run_id", None) or time.strftime("%Y%m%d_%H%M%S")
        dirpath = os.path.join("checkpoints", run_id)
        return cls(dirpath, logger)


class TimeDeltaCheckpoint(Callback):
    """
    Checkpoint callback with time-based adaptive intervals.

    Saves more frequently early in training, then less frequently:
    - Every 1 minute for the first 20 minutes
    - Every 5 minutes from 20 min to 2 hours
    - Every 20 minutes after 2 hours
    """

    def __init__(
        self,
        phase1_interval: float = 60,  # 1 minute
        phase1_duration: float = 1200,  # 20 minutes
        phase2_interval: float = 300,  # 5 minutes
        phase2_duration: float = 7200,  # 2 hours
        phase3_interval: float = 1200,  # 20 minutes
        monitor: str | None = "train/loss",
    ):
        super().__init__()
        self.phase1_interval = phase1_interval
        self.phase1_duration = phase1_duration
        self.phase2_interval = phase2_interval
        self.phase2_duration = phase2_duration
        self.phase3_interval = phase3_interval
        self.monitor = monitor
        self.start_time = None
        self.last_save_time = None
        self._best_score = float("inf")

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.last_save_time = self.start_time
        self.saver = CheckpointSaver.from_trainer(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        elapsed = now - self.start_time
        since_last_save = now - self.last_save_time

        if since_last_save >= self._get_interval(elapsed):
            self.saver.save(trainer, "latest.ckpt")
            self.last_save_time = now

            if self.monitor:
                score = trainer.callback_metrics.get(self.monitor)
                if score is not None and score.item() < self._best_score:
                    self._best_score = score.item()
                    self.saver.save(trainer, "best.ckpt")

    def _get_interval(self, elapsed: float) -> float:
        if elapsed < self.phase1_duration:
            return self.phase1_interval
        elif elapsed < self.phase2_duration:
            return self.phase2_interval
        else:
            return self.phase3_interval

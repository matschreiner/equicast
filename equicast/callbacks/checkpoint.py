import os
import time

from pytorch_lightning.callbacks import ModelCheckpoint


class TimeDeltaCheckpoint(ModelCheckpoint):
    """
    Checkpoint callback with time-based adaptive intervals.

    Saves more frequently early in training, then less frequently:
    - Every 1 minute for the first 20 minutes
    - Every 5 minutes from 20 min to 2 hours
    - Every 20 minutes after 2 hours
    """

    def __init__(
        self,
        dirpath: str | None = None,
        phase1_interval: float = 60,
        phase1_duration: float = 1200,
        phase2_interval: float = 300,
        phase2_duration: float = 7200,
        phase3_interval: float = 1200,
    ):
        super().__init__(
            dirpath=dirpath,
            filename="latest",
            save_top_k=-1,
            every_n_train_steps=1,
            enable_version_counter=False,
        )
        self.phase1_interval = phase1_interval
        self.phase1_duration = phase1_duration
        self.phase2_interval = phase2_interval
        self.phase2_duration = phase2_duration
        self.phase3_interval = phase3_interval
        self.start_time = None
        self.last_save_time = None

    def on_train_start(self, trainer, pl_module):
        if self.dirpath is None:
            run_id = getattr(trainer.logger, "run_id", None) or time.strftime("%Y%m%d_%H%M%S")
            self.dirpath = os.path.join("checkpoints", run_id)
        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()
        self.last_save_time = self.start_time

    def _should_skip_saving_checkpoint(self, trainer) -> bool:
        if self.start_time is None:
            return True
        now = time.time()
        elapsed = now - self.start_time
        if (now - self.last_save_time) < self._get_interval(elapsed):
            return True
        self.last_save_time = now
        return False

    def _save_checkpoint(self, trainer, filepath):
        trainer.save_checkpoint(filepath, self.save_weights_only)
        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                if hasattr(logger, "after_save_checkpoint"):
                    logger.after_save_checkpoint(filepath)

    def _get_interval(self, elapsed: float) -> float:
        if elapsed < self.phase1_duration:
            return self.phase1_interval
        elif elapsed < self.phase2_duration:
            return self.phase2_interval
        else:
            return self.phase3_interval

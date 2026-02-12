"""Custom PyTorch Lightning callbacks."""

import os
import time

from pytorch_lightning import Callback


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
        monitor: str = "train/loss_step",
        save_initial: bool = False,
    ):
        super().__init__()
        self.phase1_interval = phase1_interval
        self.phase1_duration = phase1_duration
        self.phase2_interval = phase2_interval
        self.phase2_duration = phase2_duration
        self.phase3_interval = phase3_interval
        self.monitor = monitor
        self.save_initial = save_initial

        self.start_time = None
        self.last_save_time = None
        self.best_loss = float("inf")

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.last_save_time = self.start_time

        dirpath = getattr(trainer.checkpoint_callback, "dirpath", None) or trainer.log_dir
        print(f"Checkpoint directory: {dirpath}")
        if trainer.logger and hasattr(trainer.logger, "run_id"):
            run = trainer.logger.experiment.get_run(trainer.logger.run_id)
            print(f"MLflow run: {run.info.run_name}")

        if self.save_initial:
            self._save_checkpoint(trainer, "initial")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        now = time.time()
        elapsed = now - self.start_time
        since_last_save = now - self.last_save_time

        interval = self._get_interval(elapsed)

        if since_last_save >= interval:
            self._save_checkpoint(trainer, "latest")
            self._maybe_save_best(trainer)
            self.last_save_time = now

    def _get_interval(self, elapsed: float) -> float:
        if elapsed < self.phase1_duration:
            return self.phase1_interval
        elif elapsed < self.phase2_duration:
            return self.phase2_interval
        else:
            return self.phase3_interval

    def _save_checkpoint(self, trainer, filename: str):
        dirpath = getattr(trainer.checkpoint_callback, "dirpath", None) or trainer.log_dir
        filepath = os.path.join(dirpath, f"{filename}.ckpt")
        trainer.save_checkpoint(filepath)

    def _maybe_save_best(self, trainer):
        current_loss = trainer.callback_metrics.get(self.monitor)
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            self._save_checkpoint(trainer, "best")

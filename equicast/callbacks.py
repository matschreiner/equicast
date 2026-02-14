"""Custom PyTorch Lightning callbacks."""

import copy
import os
import time

import torch
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
        if trainer.logger and hasattr(trainer.logger, "run_id") and trainer.is_global_zero:
            run = trainer.logger.experiment.get_run(trainer.logger.run_id)
            if run is not None:
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
        print(f"Saved checkpoint: {filepath}")

    def _maybe_save_best(self, trainer):
        current_loss = trainer.callback_metrics.get(self.monitor)
        if current_loss is not None and current_loss < self.best_loss:
            self.best_loss = current_loss
            self._save_checkpoint(trainer, "best")


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

import sys
from dataclasses import dataclass

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from equicast.experiment.config import ExperimentConfig
from equicast.logger import BaseLogger
from equicast.model import Model


@dataclass
class TrainConfig(ExperimentConfig):
    model: Model
    trainer: Trainer
    dataloader: DataLoader
    logger: BaseLogger
    val_dataloader: DataLoader | None = None
    experiment_name: str = "train"
    ckpt_path: str | None = None

    def run(self):
        from equicast.utils.git import get_git_hash
        self.logger.log_hyperparams({
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "command": " ".join(sys.argv),
            "git_commit": get_git_hash() or "unknown",
        })
        self.trainer.fit(
            self.model,
            self.dataloader,
            val_dataloaders=self.val_dataloader,
            ckpt_path=self.ckpt_path,
            weights_only=False,
        )

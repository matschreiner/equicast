from dataclasses import dataclass

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from equicast.experiments.config import ExperimentConfig
from equicast.logger import BaseLogger
from equicast.model import Model


@dataclass
class TrainConfig(ExperimentConfig):
    model: Model
    trainer: Trainer
    dataloader: DataLoader
    logger: BaseLogger
    experiment_name: str = "train"
    ckpt_path: str | None = None

    def run(self):
        self.trainer.fit(
            self.model,
            self.dataloader,
            ckpt_path=self.ckpt_path,
            weights_only=False,
        )

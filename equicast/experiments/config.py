import json
import traceback
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import fiddle as fdl
import torch
from anemoi.datasets import open_dataset
from fiddle import graphviz
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from equicast import cute
from equicast.data.graph_provider import BaseGraphProvider
from equicast.forecaster import Forecaster
from equicast.logger import BaseLogger
from equicast.model import Model
from equicast.utils import get_git_info


class ExperimentConfig(ABC):
    experiment_name: str
    experiment_dir: str | None = None

    @abstractmethod
    def run(self): ...


@dataclass
class TrainConfig(ExperimentConfig):
    model: Model
    trainer: Trainer
    dataloader: DataLoader
    logger: BaseLogger
    experiment_name: str = "train"

    def run(self):
        # Set trainer default_root_dir to experiment_dir if provided
        self.trainer.fit(
            self.model,
            self.dataloader,
        )


@dataclass
class ForecastConfig(ExperimentConfig):
    forecaster: Forecaster
    timeseries: torch.Tensor
    graph: Data
    logger: BaseLogger
    experiment_name: str = "forecast"

    def run(self):
        self.forecaster.forecast(
            timeseries=self.timeseries,
            graph=self.graph,
        )


def vis_config(config):
    try:
        graph = graphviz.render(config)
        graph.view()
    except Exception as _:
        pass


def run_experiment(config: fdl.Config):
    vis_config(config)
    experiment = fdl.build(config)
    metadata = get_metadata()
    experiment.logger.log_hyperparams(metadata)

    experiment.run()


def get_metadata():
    git_info = get_git_info()
    run_name = cute()
    timestamp = datetime.now()

    metadata = {
        "run_name": run_name,
        "timestamp": timestamp.isoformat(),
        "git_hash": git_info.get("hash"),
        "git_short_hash": git_info.get("short_hash"),
        "git_branch": git_info.get("branch"),
        "git_dirty": git_info.get("dirty"),
    }

    return metadata

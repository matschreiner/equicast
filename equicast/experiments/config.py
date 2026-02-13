import argparse
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
    logger: BaseLogger
    experiment_name: str = "forecast"

    def run(self):
        self.forecaster.forecast(
            timeseries=self.timeseries,
        )


def vis_config(config):
    try:
        graph = graphviz.render(config)
        graph.view()
    except Exception as _:
        pass


def run_experiment(config: fdl.Config):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph", action="store_true", help="Open graphviz visualization of the config"
    )
    args, _ = parser.parse_known_args()
    if args.graph:
        vis_config(config)
    experiment = fdl.build(config)
    experiment.run()

    #  git_info = get_git_info()
    #  experiment.logger.log_hyperparams(git_info)
    #
    #  with experiment_error_handler(experiment.logger):


@contextmanager
def experiment_error_handler(logger: BaseLogger):
    logger.log_hyperparams({"status": "started"})
    try:
        yield
        logger.log_hyperparams({"status": "completed"})
    except KeyboardInterrupt:
        logger.log_hyperparams({"status": "interrupted"})
        raise
    except Exception as e:
        logger.log_hyperparams(
            {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        )
        raise

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import traceback

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
from equicast.model import Model


class ExperimentConfig(ABC):
    experiment_name: str
    experiment_dir: str | None = None

    @abstractmethod
    def run(self): ...

    def set_experiment_dir(self, experiment_dir: str):
        """Set the experiment directory for this experiment."""
        self.experiment_dir = experiment_dir


@dataclass
class TrainConfig(ExperimentConfig):
    model: Model
    trainer: Trainer
    dataloader: DataLoader
    logger: Logger
    experiment_name: str = "train"

    def run(self):
        # Set trainer default_root_dir to experiment_dir if provided
        if self.experiment_dir:
            self.trainer.default_root_dir = self.experiment_dir

        self.trainer.fit(
            self.model,
            self.dataloader,
        )


@dataclass
class ForecastConfig(ExperimentConfig):
    forecaster: Forecaster
    timeseries: torch.Tensor
    graph: Data
    experiment_name: str = "forecast"

    def run(self):
        # Use experiment_dir if provided, otherwise use current directory
        output_dir = self.experiment_dir if self.experiment_dir else "."
        self.forecaster.forecast(
            timeseries=self.timeseries,
            graph=self.graph,
            output_dir=output_dir,
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

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = Path("experiments") / experiment.experiment_name / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Set the experiment directory on the experiment
    experiment.set_experiment_dir(str(experiment_dir))

    print(f"Running experiment in: {experiment_dir}")
    experiment.run()

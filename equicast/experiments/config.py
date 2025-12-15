from abc import ABC, abstractmethod
from dataclasses import dataclass

import fiddle as fdl
import torch
from anemoi.datasets import open_dataset
from fiddle import graphviz
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from equicast.forecaster import Forecaster
from equicast.graph.graph_provider import BaseGraphProvider
from equicast.model import Model


class ExperimentConfig(ABC):
    @abstractmethod
    def run(self): ...


@dataclass
class TrainConfig(ExperimentConfig):
    model: Model
    trainer: Trainer
    dataloader: DataLoader
    logger: Logger

    def run(self):
        self.trainer.fit(
            self.model,
            self.dataloader,
        )


@dataclass
class ForecastConfig(ExperimentConfig):
    forecaster: Forecaster
    timeseries: torch.Tensor
    graph: Data

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
    experiment.run()

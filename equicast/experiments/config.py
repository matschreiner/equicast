from abc import ABC, abstractmethod
from dataclasses import dataclass

import fiddle as fdl
from fiddle import graphviz
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.logger import Logger
from torch.utils.data import DataLoader

from equicast.forecaster import Forecaster
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
    dataset_path: str
    graph_provider: any  # BaseGraphProvider
    start_idx: int
    steps: int

    def run(self):
        import torch
        from anemoi.datasets import open_dataset

        # Open dataset and load timeseries
        data = open_dataset(self.dataset_path)
        timeseries = (
            torch.tensor(data[self.start_idx : self.start_idx + 1 + self.steps])
            .squeeze()
            .permute(0, 2, 1)
        )  # [time, nodes, features]

        # Create initial state graph from first timestep
        initial_graph = self.graph_provider.get_graph()
        initial_graph["grid"].input_state = timeseries[0]

        # Extract forcing sequence for timesteps 1 to steps
        data_handler = self.forecaster.model.data_handler
        forcing_idxs = data_handler.feature_router._get_data_idxs(
            data_handler.feature_router.feature_config.forcing
        )
        forcing_sequence = [
            timeseries[i][:, forcing_idxs] for i in range(1, self.steps + 1)
        ]

        # Run forecast
        predictions = self.forecaster.forecast(
            initial_graph,
            steps=self.steps,
            forcing_sequence=forcing_sequence,
        )

        return predictions


def vis_config(config):
    graph = graphviz.render(config)
    graph.view()


def run_experiment(config: fdl.Config):
    vis_config(config)
    experiment = fdl.build(config)
    experiment.run()

import argparse
import importlib
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
from equicast.utils import get_git_info, get_hardware_info


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
    ckpt_path: str | None = None

    def run(self):
        self.trainer.fit(
            self.model,
            self.dataloader,
            ckpt_path=self.ckpt_path,
            weights_only=False,
        )


@dataclass
class ForecastConfig(ExperimentConfig):
    forecaster: Forecaster
    timeseries: torch.Tensor
    logger: BaseLogger
    model_id: str = ""
    experiment_name: str = "forecast"

    def run(self):
        output_dir = f"forecasts/{self.model_id}"
        self.forecaster.forecast(timeseries=self.timeseries, output_dir=output_dir)


def vis_config(config):
    try:
        graph = graphviz.render(config)
        graph.view()
    except Exception as _:
        pass


def _apply_fiddler(config: fdl.Config, fiddler_spec: str):
    """Apply a fiddler to the config. Format: 'name' or 'name:arg'."""
    if ":" in fiddler_spec:
        name, arg = fiddler_spec.split(":", 1)
    else:
        name, arg = fiddler_spec, None

    module = importlib.import_module(f"config.fiddlers.{name}")
    if arg is not None:
        module.fiddler(config, arg)
    else:
        module.fiddler(config)


def run_experiment(config: fdl.Config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="store_true", help="Open graphviz visualization of the config")
    parser.add_argument("--fiddler", action="append", default=[], help="Fiddlers (repeatable, use name:arg for args)")
    args, _ = parser.parse_known_args()

    if args.graph:
        vis_config(config)

    for fiddler_spec in args.fiddler:
        _apply_fiddler(config, fiddler_spec)

    experiment = fdl.build(config)
    experiment.logger.log_hyperparams(get_git_info())
    experiment.logger.log_hyperparams(get_hardware_info())
    experiment.run()


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

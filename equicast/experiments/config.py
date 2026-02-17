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
    experiment.logger.log_hyperparams({"num_parameters": sum(p.numel() for p in experiment.model.parameters())})
    _log_config(experiment.logger, config)
    experiment.run()


def flatten_config(config: fdl.Config) -> dict[str, any]:
    """Flatten a fiddle config into a dict with dot-separated paths."""
    from fiddle import daglish

    flat = {}
    for value, path in daglish.iterate(config):
        if isinstance(value, fdl.Config):
            continue
        key = daglish.path_str(path).lstrip(".")
        flat[key] = value
    return flat


def _log_config(logger, config: fdl.Config):
    """Log the fiddle config as a YAML artifact and flat hyperparams."""
    import tempfile

    from fiddle.experimental.yaml_serialization import dump_yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="config_") as f:
        f.write(dump_yaml(config))
        f.flush()
        logger.log_artifact(f.name, artifact_path="")

    flat_config = flatten_config(config)
    for k, w in flat_config.items():
        logger.log_hyperparams({k: w})

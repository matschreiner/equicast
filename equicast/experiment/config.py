import argparse
import importlib
import tempfile
from abc import ABC, abstractmethod

import fiddle as fdl
from fiddle import daglish, graphviz
from fiddle.experimental.yaml_serialization import dump_yaml

from equicast.utils import get_git_info, get_hardware_info


class ExperimentConfig(ABC):
    experiment_name: str
    experiment_dir: str | None = None

    @abstractmethod
    def run(self): ...


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
    if hasattr(experiment.logger, "experiment_name"):
        fix_artifact_location(experiment.logger.experiment_name)
    experiment.logger.log_hyperparams(get_git_info())
    experiment.logger.log_hyperparams(get_hardware_info())
    _log_config(experiment.logger, config)
    experiment.run()


def flatten_config(config: fdl.Config) -> dict[str, any]:
    """Flatten a fiddle config into a dict with dot-separated paths."""

    flat = {}
    for value, path in daglish.iterate(config):
        if isinstance(value, fdl.Config):
            continue
        key = daglish.path_str(path).lstrip(".")
        flat[key] = value
    return flat


def _log_config(logger, config: fdl.Config):
    """Log the fiddle config as a YAML artifact and flat hyperparams."""

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="config_") as f:
            f.write(dump_yaml(config))
            f.flush()
            logger.log_artifact(f.name, artifact_path="")
    except Exception:
        pass

    flat_config = flatten_config(config)
    for k, w in flat_config.items():
        logger.log_hyperparams({k: w})

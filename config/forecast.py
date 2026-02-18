"""Forecast from a trained model checkpoint.

Usage:
    python config/forecast.py <run_name>
"""

import sys

import fiddle as fdl

from equicast import data, experiment
from equicast.checkpoint import MLFlowCheckpointProvider
from equicast.forecaster import Forecaster
from equicast.logger import MLFlowLogger
from equicast.model import Model
from equicast.model.from_checkpoint import from_checkpoint

DATASET_PATH = "/home/masc/storage/era5-o96-2024-tail200-6h.zarr"
GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"


def default_logger():
    return fdl.Config(MLFlowLogger, experiment_name="equicast")


def default_model(run_id):
    checkpoint_provider = fdl.Config(
        MLFlowCheckpointProvider,
        run_id=run_id,
        checkpoint_name="latest",
    )
    return fdl.Config(load_model, model_cls=Model, checkpoint_provider=checkpoint_provider)


def default_forecaster(model, logger):
    return fdl.Config(Forecaster, model=model, logger=logger)


def default_data_handler(model):
    return fdl.Config(get_data_handler, model)


def default_dataset(dataset_path, graph_path):
    graph_provider = fdl.Config(data.StaticGraphProvider, graph_path=graph_path)
    return fdl.Config(data.AnemoiDataset, path=dataset_path, graph_provider=graph_provider, subsample=1)


def main():
    run_name = sys.argv[1]
    run_id = resolve_run(run_name)

    logger = default_logger()
    model = default_model(run_id)
    dataset = default_dataset(DATASET_PATH, GRAPH_PATH)

    cfg = fdl.Config(
        experiment.ForecastConfig,
        forecaster=default_forecaster(model, logger),
        input_timeseries=fdl.Config(get_input_timeseries, dataset=dataset),
        target_timeseries=fdl.Config(get_target_timeseries, dataset=dataset),
        data_handler=default_data_handler(model),
        logger=logger,
        model_id=run_name,
    )

    experiment.run_experiment(cfg)


def load_model(model_cls, checkpoint_provider):
    return from_checkpoint(model_cls, checkpoint_provider)


def get_data_handler(model):
    return model.data_handler


def get_input_timeseries(dataset, num_samples=50):
    return [dataset[i]["input"] for i in range(num_samples)]


def get_target_timeseries(dataset, num_samples=50):
    return [dataset[i]["target"] for i in range(num_samples)]


def resolve_run(run_name: str) -> str:
    import mlflow
    runs = mlflow.search_runs(
        filter_string=f"attributes.run_name = '{run_name}'",
        search_all_experiments=True,
    )
    if runs.empty:
        raise ValueError(f"No run found with name '{run_name}'")
    return runs.iloc[0]["run_id"]


if __name__ == "__main__":
    main()

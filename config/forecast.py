"""Forecast from a trained model checkpoint."""

import fiddle as fdl

from equicast import experiment
from equicast.checkpoint import RsyncCheckpointProvider
from equicast.data.dataset.dataset import AnemoiDataset
from equicast.data.graph_provider import StaticGraphProvider
from equicast.forecaster import Forecaster
from equicast.logger import MLFlowLogger
from equicast.model import Model
from equicast.model.from_checkpoint import from_checkpoint


def load_model(model_cls, checkpoint_provider):
    return from_checkpoint(model_cls, checkpoint_provider)


def get_data_handler(model):
    return model.data_handler


def get_timeseries(dataset, num_samples=10):
    input_timeseries = [dataset[i]["input"] for i in range(num_samples)]
    target_timeseries = [dataset[i]["target"] for i in range(num_samples)]
    return input_timeseries, target_timeseries


def main():
    dataset_path = "/home/masc/storage/era5-o96-2024-tail200-6h.zarr"
    #  Scheduler
    checkpoint_path = "/leonardo/home/userexternal/jschrei1/equicast/mlruns/394873961037717851/80a28951aa9941578184d78d870e663e/checkpoints/latest.ckpt"
    # EMA
    #  checkpoint_path = "/leonardo/home/userexternal/jschrei1/equicast/mlruns/394873961037717851/bbfe37a6679f490abf1487be55124d16/checkpoints/latest.ckpt"
    # VANILLA
    #  checkpoint_path = "/leonardo/home/userexternal/jschrei1/equicast/mlruns/394873961037717851/a7004148bd5a4a2fae6062725d8dc090/checkpoints/latest.ckpt"

    model_id = checkpoint_path.split("/")[-3]  # MLflow run ID
    host = "leonardo"

    graph_provider = StaticGraphProvider(graph_path="graph/aifs-graphcast-unnormed.pt")
    dataset = AnemoiDataset(path=dataset_path, graph_provider=graph_provider, subsample=1)

    input_timeseries, target_timeseries = get_timeseries(dataset)

    logger = fdl.Config(
        MLFlowLogger,
        experiment_name="masc1",
        tracking_uri="https://mlflow.dmidev.org/",
    )

    checkpoint_provider = fdl.Config(
        RsyncCheckpointProvider,
        remote_path=checkpoint_path,
        host=host,
    )

    model = fdl.Config(
        load_model,
        model_cls=Model,
        checkpoint_provider=checkpoint_provider,
    )

    data_handler = fdl.Config(
        get_data_handler,
        model,
    )

    forecaster = fdl.Config(
        Forecaster,
        model=model,
        logger=logger,
    )

    cfg = fdl.Config(
        experiment.ForecastConfig,
        forecaster=forecaster,
        input_timeseries=input_timeseries,
        target_timeseries=target_timeseries,
        data_handler=data_handler,
        logger=logger,
        model_id=model_id,
    )

    experiment.run_experiment(cfg)


if __name__ == "__main__":
    main()

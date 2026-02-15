"""Forecast from a trained model checkpoint."""

import fiddle as fdl

from equicast import experiments
from equicast.checkpoint import RsyncCheckpointProvider
from equicast.data.dataset.dataset import AnemoiDataset
from equicast.forecaster import Forecaster
from equicast.logger import MLFlowLogger
from equicast.model import Model
from equicast.model.from_checkpoint import from_checkpoint


def load_model(model_cls, checkpoint_provider):
    return from_checkpoint(model_cls, checkpoint_provider)


def main():
    dataset_path = "/home/masc/storage/era5-o96-2024-tail200-6h.zarr"
    checkpoint_path = "/leonardo/home/userexternal/jschrei1/equicast/mlruns/394873961037717851/80a28951aa9941578184d78d870e663e/checkpoints/latest.ckpt"
    host = "leonardo"

    dataset = AnemoiDataset(path=dataset_path, graph_provider=None, subsample=1)
    timeseries = [dataset[i]["input"] for i in range(50)]

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

    forecaster = fdl.Config(
        Forecaster,
        model=model,
        logger=logger,
    )

    cfg = fdl.Config(
        experiments.ForecastConfig,
        forecaster=forecaster,
        timeseries=timeseries,
        logger=logger,
    )

    experiments.run_experiment(cfg)


if __name__ == "__main__":
    main()

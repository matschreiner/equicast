"""Forecast from a trained model checkpoint."""

import fiddle as fdl
import torch
from anemoi.datasets import open_dataset

from equicast import data, experiments
from equicast.checkpoint import RemoteCheckpointProvider
from equicast.forecaster import Forecaster
from equicast.logger import CSVLogger, MLFlowLogger
from equicast.model import Model
from equicast.model.backbones.gnn import GNN
from equicast.model.from_checkpoint import from_checkpoint


def get_graph_from_provider(graph_provider, idx):
    """Get graph data from graph provider."""
    graph = graph_provider.get_graph(idx)
    return graph


def get_timeseries(
    dataset_path: str,
    start_idx,
    steps,
) -> torch.Tensor:
    """Get timeseries data from dataset for forecasting."""
    data = open_dataset(dataset_path)
    timeseries = (
        torch.tensor(data[start_idx : start_idx + 1 + steps])
        .squeeze()
        .permute(0, 2, 1)
    )
    return timeseries


def main():
    timeseries = fdl.Config(
        get_timeseries,
        dataset_path="/home/masc/storage/mini_aifs.zarr",
        start_idx=0,
        steps=100,
    )

    logger = fdl.Config(
        MLFlowLogger,
        experiment_name="masc",
        tracking_uri="https://mlflow.dmidev.org/",
    )

    graph_provider = fdl.Config(
        data.StaticGraphProvider,
        path="./graph/aifs-graphcast.pt",
    )
    graph = fdl.Config(
        get_graph_from_provider,
        graph_provider=graph_provider,
        idx=None,
    )

    checkpoint_provider = fdl.Config(
        RemoteCheckpointProvider,
        remote_path="/vf/masc/programming/equicast/26/ad5bc3e47a4046d0b98db1bd527cf00b/checkpoints/latest.ckpt",
        host="ohm",
    )

    model = fdl.Config(
        from_checkpoint,
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
        graph=graph,
        timeseries=timeseries,
        logger=logger,
    )

    experiments.run_experiment(cfg)


if __name__ == "__main__":
    main()

"""Forecast from a trained model checkpoint."""

import base_config
import fiddle as fdl
import torch
from anemoi.datasets import open_dataset

from config.base_config import (
    get_data_handler,
    get_feature_config,
    get_forecaster,
)
from equicast import experiments
from equicast.model import Model
from equicast.model.backbones.gnn import GNN


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
    feature_config = get_feature_config()
    data_handler = get_data_handler(feature_config=feature_config)

    timeseries = fdl.Config(
        get_timeseries,
        dataset_path="/home/masc/storage/mini_aifs.zarr",
        start_idx=0,
        steps=10,
    )
    backbone = fdl.Config(
        GNN,
        feature_config,
    )

    graph_provider = base_config.get_graph_provider()
    graph = fdl.Config(
        get_graph_from_provider,
        graph_provider=graph_provider,
        idx=None,
    )

    model = fdl.Config(
        Model,
        backbone=backbone,
        data_handler=data_handler,
        optimizer_factory=None,
        scheduler_factory=None,
    )
    forecaster = get_forecaster(model)

    cfg = fdl.Config(
        experiments.ForecastConfig,
        forecaster=forecaster,
        graph=graph,
        timeseries=timeseries,
    )

    experiments.run_experiment(cfg)


if __name__ == "__main__":
    main()

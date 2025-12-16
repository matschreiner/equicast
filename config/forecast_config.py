"""Forecast from a trained model checkpoint."""

import fiddle as fdl
import torch
from anemoi.datasets import open_dataset

from equicast import data, experiments
from equicast.forecaster import Forecaster
from equicast.model import Model
from equicast.model.backbones.gnn import GNN
from equicast.utils.mlflow_loader import load_model_from_mlflow


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
        steps=10,
    )

    graph_provider = fdl.Config(
        data.StaticGraphProvider,
        path="./graph/aifs-single.pt",
    )
    graph = fdl.Config(
        get_graph_from_provider,
        graph_provider=graph_provider,
        idx=None,
    )

    #  Commented out: instantiating new model
    #
    #  feature_config = get_feature_config()
    #  data_handler = get_data_handler(feature_config=feature_config)
    #  backbone = fdl.Config(
    #      GNN,
    #      feature_config,
    #  )
    #
    # model = fdl.Config(
    #     Model,
    #     backbone=backbone,
    #     data_handler=data_handler,
    #     optimizer_factory=None,
    #     scheduler_factory=None,
    # )

    model = fdl.Config(
        load_model_from_mlflow,
        run_id="ed5d0337880c4c6684ba4991ff36d5c9",
        checkpoint_name=None,
    )

    forecaster = fdl.Config(
        Forecaster,
        model=model,
    )

    cfg = fdl.Config(
        experiments.ForecastConfig,
        forecaster=forecaster,
        graph=graph,
        timeseries=timeseries,
    )

    experiments.run_experiment(cfg)


if __name__ == "__main__":
    main()

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
    dataset,
    start_idx,
    steps,
):
    """Get timeseries data from dataset for forecasting."""

    timeseries = [
        dataset[idx] for idx in range(start_idx, start_idx + steps + 1)
    ]
    return timeseries


def main():
    dataset_path = "/home/masc/storage/mini_aifs.zarr"
    graph_path = "graph/aifs-single.pt"

    graph_provider = fdl.Config(
        data.StaticGraphProvider,
        path=graph_path,
    )
    dataset = fdl.Config(
        data.AnemoiDataset,
        path=dataset_path,
        graph_provider=graph_provider,
    )

    timeseries = fdl.Config(
        get_timeseries,
        dataset=dataset,
        start_idx=0,
        steps=109,
    )

    logger = fdl.Config(
        MLFlowLogger,
        experiment_name="masc",
        tracking_uri="https://mlflow.dmidev.org/",
    )
    #  graph_provider = fdl.Config(
    #      data.StaticGraphProvider,
    #      path="./graph/aifs-single.pt",
    #  )
    #  graph = fdl.Config(
    #      get_graph_from_provider,
    #      graph_provider=graph_provider,
    #      idx=None,
    #  )

    #  checkpoint_provider = fdl.Config(
    #      RemoteCheckpointProvider,
    #      remote_path="/vf/masc/programming/equicast/26/ad5bc3e47a4046d0b98db1bd527cf00b/checkpoints/latest.ckpt",
    #      host="ohm",
    #  )
    #  model = fdl.Config(
    #      from_checkpoint,
    #      model_cls=Model,
    #      checkpoint_provider=checkpoint_provider,
    #  )

    model = get_gnn()

    forecaster = fdl.Config(
        Forecaster,
        model=model,
        logger=logger,
    )

    cfg = fdl.Config(
        experiments.ForecastConfig,
        forecaster=forecaster,
        #  graph=graph,
        timeseries=timeseries,
        logger=logger,
    )

    experiments.run_experiment(cfg)


def get_gnn():
    feature_config = data.FeatureConfig.from_yaml(
        path="hydraconfig/features/base.yaml",
    )
    model = fdl.Config(
        Model,
        backbone=fdl.Config(
            GNN,
            feature_config=feature_config,
        ),
        data_handler=fdl.Config(
            data.GraphDataHandler,
            feature_config=feature_config,
            dataset_path="/home/masc/storage/mini_aifs.zarr",
        ),
    )
    return model


if __name__ == "__main__":
    main()

"""Forecast from a trained model checkpoint."""

import fiddle as fdl
import torch
from anemoi.datasets import open_dataset

from equicast import data, experiments
from equicast.checkpoint import (
    LocalCheckpointProvider,
    MLFlowCheckpointProvider,
    RemoteCheckpointProvider,
    RsyncCheckpointProvider,
)
from equicast.forecaster import Forecaster
from equicast.logger import CSVLogger, MLFlowLogger
from equicast.model import Model
from equicast.model.backbones.gnn import GNN
from equicast.model.from_checkpoint import from_checkpoint
from equicast.utils.mlflow_loader import load_model_from_mlflow


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

    timeseries = [dataset[idx] for idx in range(start_idx, start_idx + steps + 1)]
    return timeseries


def main():
    dataset_path = "/home/masc/storage/mini_aifs.zarr"
    graph_path = "graph/aifs-graphcast.pt"
    checkpoint_path = (
        "/leonardo/home/userexternal/jschrei1/equicast/logs/equicast/version_8/checkpoints/latest.ckpt"
        #  "/vf/masc/programming/equicast/58/fa0e587422c24628b1d04541bc02e084/checkpoints/latest.ckpt"
    )
    host = "leonardo"

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
        steps=56,
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
    #      remote_path="/vf/masc/programming/equicast/58/7122bdee98804e149b1ed606a598c4a3/checkpoints/latest.ckpt",
    #      host="ohm",
    #  )
    #
    #  checkpoint_provider = fdl.Config(
    #      MLFlowCheckpointProvider,
    #      tracking_uri="https://mlflow.dmidev.org/",
    #      run_id="6c273433894c4f44b9613d7816b71890",
    #      checkpoint_name="latest",
    #  )
    #  checkpoint_provider = fdl.Config(
    #      LocalCheckpointProvider,
    #      checkpoint_path="latest2.ckpt",
    #  )
    #
    checkpoint_provider = fdl.Config(
        RsyncCheckpointProvider,
        remote_path=checkpoint_path,
        host=host,
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

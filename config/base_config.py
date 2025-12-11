"""Base configuration components shared across experiments."""

import fiddle as fdl

from equicast.checkpoint.checkpoint_provider import MLFlowCheckpointProvider
from equicast.data import FeatureConfig
from equicast.data.data_handler import DataHandler
from equicast.dataset import AnemoiDataset
from equicast.forecaster import Forecaster
from equicast.graph.graph_provider import StaticGraphProvider
from equicast.model import Model
from equicast.model.from_checkpoint import from_checkpoint


def get_feature_config(path: str = "hydraconfig/features/base.yaml"):
    """Get feature configuration from YAML file."""
    return fdl.Config(FeatureConfig.from_yaml, path=path)


def get_graph_provider(path: str = "./graph/aifs-single.pt"):
    """Get static graph provider configuration."""
    return fdl.Config(StaticGraphProvider, path=path)


def get_dataset(
    path: str = "/home/masc/storage/mini_aifs.zarr",
    graph_provider=None,
):
    """Get dataset configuration. Dataset returns raw data without preprocessing."""
    if graph_provider is None:
        graph_provider = get_graph_provider()

    return fdl.Config(
        AnemoiDataset,
        path=path,
        graph_provider=graph_provider,
    )


def get_data_handler(
    dataset_path: str = "/home/masc/storage/mini_aifs.zarr",
    feature_config=None,
):
    """
    Get data handler configuration for metadata management.

    DataHandler extracts statistics and feature mappings from the dataset
    without loading the full dataset. Used by forecaster for scaling and routing.
    """
    if feature_config is None:
        feature_config = get_feature_config()

    return fdl.Config(
        DataHandler,
        dataset_path=dataset_path,
        feature_config=feature_config,
    )


def get_checkpoint_provider(
    tracking_uri: str = "https://mlflow.dmidev.org/",
    run_id: str = None,
    checkpoint_name: str = "latest",
):
    """Get MLFlow checkpoint provider configuration."""
    return fdl.Config(
        MLFlowCheckpointProvider,
        tracking_uri=tracking_uri,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
    )


def get_model_from_checkpoint(checkpoint_provider=None):
    """Get model loaded from checkpoint."""
    if checkpoint_provider is None:
        checkpoint_provider = get_checkpoint_provider()

    return fdl.Config(
        from_checkpoint,
        model_cls=Model,
        checkpoint_provider=checkpoint_provider,
    )


def get_fresh_model(backbone, data_handler=None):
    """Get fresh untrained model (useful for debugging)."""
    if data_handler is None:
        data_handler = get_data_handler()

    return fdl.Config(
        Model,
        backbone=backbone,
        data_handler=data_handler,
    )


def get_forecaster(model=None):
    """Get forecaster configuration."""
    if model is None:
        model = get_model_from_checkpoint()

    return fdl.Config(
        Forecaster,
        model=model,
    )

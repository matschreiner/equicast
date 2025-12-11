"""Base configuration components shared across experiments."""

import fiddle as fdl

from equicast.data import FeatureConfig
from equicast.data.data_handler import DataHandler
from equicast.dataset import AnemoiDataset
from equicast.graph.graph_provider import StaticGraphProvider


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

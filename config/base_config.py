"""Base configuration components shared across experiments."""

import fiddle as fdl

from equicast.data import FeatureConfig
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
    feature_config=None,
    graph_provider=None,
):
    """Get dataset configuration with default feature config and graph provider."""
    if feature_config is None:
        feature_config = get_feature_config()
    if graph_provider is None:
        graph_provider = get_graph_provider()

    return fdl.Config(
        AnemoiDataset,
        path=path,
        feature_config=feature_config,
        graph_provider=graph_provider,
    )

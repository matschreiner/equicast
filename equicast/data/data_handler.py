"""DataHandler for managing scaling and feature routing metadata."""

import torch
from anemoi.datasets import open_dataset

from equicast.data.feature_router import FeatureRouter
from equicast.data.scaler import Scaler
from equicast.utils.utils import cast_dict


class DataHandler:
    """
    Lightweight handler for dataset metadata (statistics, feature mappings).

    Extracts only the metadata needed for scaling and feature routing,
    without loading the full dataset into memory.
    """

    def __init__(self, dataset_path: str, feature_config):
        """
        Initialize DataHandler from dataset path and feature configuration.

        Args:
            dataset_path: Path to the anemoi dataset (zarr file)
            feature_config: FeatureConfig with forcing, prognostic, diagnostic fields
        """
        self.dataset_path = dataset_path
        self.feature_config = feature_config

        # Open dataset to extract metadata only
        data = open_dataset(dataset_path)
        self.statistics = cast_dict(data.statistics, torch.Tensor)
        self.name_to_index = data.name_to_index

        # Initialize scaler and feature router
        self.scaler = Scaler(self.statistics)
        self.feature_router = FeatureRouter(
            feature_config=feature_config,
            name_to_index=self.name_to_index,
        )

    @property
    def in_idxs(self):
        """Input feature indices (forcing + prognostic)."""
        return self.feature_router.in_idxs

    @property
    def out_idxs(self):
        """Output feature indices (prognostic + diagnostic)."""
        return self.feature_router.out_idxs

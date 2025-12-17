"""DataHandler for managing scaling and feature routing metadata."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from anemoi.datasets import open_dataset
from torch_geometric.data import Data

from equicast.data.feature_config import FeatureConfig
from equicast.data.feature_indices import FeatureIndices
from equicast.data.scaler import Scaler
from equicast.utils.utils import cast_dict


class BaseDataHandler(torch.nn.Module, ABC):
    """Standard DataHandler with z-score normalization."""

    def __init__(self, dataset_path: str, feature_config: FeatureConfig):
        super().__init__()
        data = open_dataset(dataset_path)
        statistics: dict[str, torch.Tensor] = cast_dict(
            data.statistics, torch.Tensor
        )

        self.name_to_index: dict[str, int] = data.name_to_index

        self.scaler = Scaler(statistics)
        self.feature_indices = FeatureIndices(
            feature_config=feature_config,
            name_to_index=self.name_to_index,
        )

    @property
    def in_idxs(self) -> list[int]:
        return self.feature_indices.in_idxs

    @property
    def out_idxs(self) -> list[int]:
        return self.feature_indices.out_idxs

    @abstractmethod
    def prepare_input(self, data: Data) -> Data: ...

    @abstractmethod
    def get_target(self, data: Data) -> torch.Tensor: ...

    def scale_input_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """Scale input features using only input feature statistics."""
        return self.scaler.transform_indices(input_features, self.in_idxs)

    def inverse_scale_input_features(
        self, input_features: torch.Tensor
    ) -> torch.Tensor:
        """Inverse scale input features using only input feature statistics."""
        return self.scaler.inverse_transform_indices(
            input_features, self.in_idxs
        )

    def scale_output_features(
        self, output_features: torch.Tensor
    ) -> torch.Tensor:
        """Scale output features using only output feature statistics."""
        return self.scaler.transform_indices(output_features, self.out_idxs)

    def inverse_scale_output_features(
        self, output_features: torch.Tensor
    ) -> torch.Tensor:
        """Inverse scale output features using only output feature statistics."""
        return self.scaler.inverse_transform_indices(
            output_features, self.out_idxs
        )

    def scale_features(self, features: torch.Tensor) -> torch.Tensor:
        """Scale features using all feature statistics."""
        return self.scaler.transform(features)

    def inverse_scale_features(self, features: torch.Tensor) -> torch.Tensor:
        """Inverse scale features using all feature statistics."""
        return self.scaler.inverse_transform(features)

    def get_input_features(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., self.in_idxs]

    def get_output_features(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., self.out_idxs]

    def pad_input_features(self, input_features: torch.Tensor) -> torch.Tensor:
        *batch_dims, _ = input_features.shape
        total_features = len(self.name_to_index)

        padded = torch.zeros(
            *batch_dims,
            total_features,
            device=input_features.device,
            dtype=input_features.dtype,
        )
        padded[..., self.in_idxs] = input_features
        return padded

    def pad_output_features(self, output_features: torch.Tensor) -> torch.Tensor:
        *batch_dims, _ = output_features.shape
        total_features = len(self.name_to_index)

        padded = torch.zeros(
            *batch_dims,
            total_features,
            device=output_features.device,
            dtype=output_features.dtype,
        )
        padded[..., self.out_idxs] = output_features
        return padded

"""DataHandler for managing scaling and feature routing metadata."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from anemoi.datasets import open_dataset
from torch_geometric.data import Data

from equicast.data.feature_config import FeatureConfig
from equicast.data.feature_indices import FeatureIndices
from equicast.data.normalizer import Normalizer
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

        self.normalizer = Normalizer(statistics)
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
    def update_next_with_prediction(
        self, next: Any, pred: torch.Tensor
    ) -> Any: ...

    @abstractmethod
    def prepare_input(self, data: Any) -> Any: ...

    @abstractmethod
    def get_target(self, data: Any) -> torch.Tensor: ...

    def normalize_input_features(
        self, input_features: torch.Tensor
    ) -> torch.Tensor:
        """Scale input features using only input feature statistics."""
        return self.normalizer.transform_indices(input_features, self.in_idxs)

    def inverse_normalize_input_features(
        self, input_features: torch.Tensor
    ) -> torch.Tensor:
        """Inverse normalize input features using only input feature statistics."""
        return self.normalizer.inverse_transform_indices(
            input_features, self.in_idxs
        )

    def normalize_output_features(
        self, output_features: torch.Tensor
    ) -> torch.Tensor:
        """Scale output features using only output feature statistics."""
        return self.normalizer.transform_indices(output_features, self.out_idxs)

    def inverse_normalize_output_features(
        self, output_features: torch.Tensor
    ) -> torch.Tensor:
        """Inverse normalize output features using only output feature statistics."""
        return self.normalizer.inverse_transform_indices(
            output_features, self.out_idxs
        )

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Scale features using all feature statistics."""
        return self.normalizer.transform(features)

    def inverse_normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Inverse normalize features using all feature statistics."""
        return self.normalizer.inverse_transform(features)

    def get_input_features(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., self.in_idxs]

    def get_output_features(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., self.out_idxs]

    def pad_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Pad prediction to full feature size.

        Args:
            prediction: Output features (prognostic + diagnostic)
                       Shape: [..., len(out_idxs)]

        Returns:
            Padded tensor with prognostic placed at correct indices
            Shape: [..., total_features]
        """
        *batch_dims, _ = prediction.shape
        total_features = len(self.name_to_index)

        padded = torch.zeros(
            *batch_dims,
            total_features,
            device=prediction.device,
            dtype=prediction.dtype,
        )

        num_prognostic = len(self.feature_indices.prognostic_idxs)
        padded[..., self.feature_indices.prognostic_idxs] = prediction[
            ..., :num_prognostic
        ]

        return padded


class GraphDataHandler(BaseDataHandler):
    """DataHandler for graph data."""

    def prepare_input(self, data: Data) -> Data:
        normalized = self.normalize_features(data["grid"].raw_input)
        data["grid"]["input"] = self.get_input_features(normalized)
        data["grid"]["residual"] = self.get_output_features(normalized)

        return data

    def get_target(self, data: Data) -> torch.Tensor:
        normalized = self.normalize_features(data["grid"].raw_target)
        target = self.get_output_features(normalized)
        return target

    def update_next_with_prediction(
        self, data: Data, pred: torch.Tensor
    ) -> Data:
        """
        Create next state from prediction + forcing.

        Args:
            data: Next graph with raw_target (for forcing)
            pred: Current prediction (output features, denormalized)

        Returns:
            Updated graph with raw_input set
        """
        next_state = self.pad_prediction(pred)
        forcing = data["grid"].raw_target[..., self.feature_indices.forcing_idxs]
        next_state[..., self.feature_indices.forcing_idxs] = forcing

        data["grid"].raw_input = next_state

        return data

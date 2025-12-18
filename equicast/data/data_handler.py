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

    def get_forcing_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract forcing features from full state."""
        return tensor[..., self.feature_indices.forcing_idxs]

    def get_prognostic_features(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract prognostic features from full state."""
        return tensor[..., self.feature_indices.prognostic_idxs]

    def update_state_with_prediction(
        self,
        prediction: torch.Tensor,
        forcing: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create next state by combining prediction with forcing.

        Args:
            prediction: Output features (prognostic + diagnostic) in normalized space
                       Shape: [..., len(out_idxs)]
            forcing: Forcing features in normalized space
                    Shape: [..., len(forcing_idxs)]

        Returns:
            Next state with all features (normalized)
            Shape: [..., total_features]
        """
        *batch_dims, _ = prediction.shape
        total_features = len(self.name_to_index)

        # Create new state tensor
        new_state = torch.zeros(
            *batch_dims,
            total_features,
            device=prediction.device,
            dtype=prediction.dtype,
        )

        # Place forcing variables
        new_state[..., self.feature_indices.forcing_idxs] = forcing

        # Extract and place prognostic from prediction
        # Prediction contains [prognostic, diagnostic], we only need prognostic
        num_prognostic = len(self.feature_indices.prognostic_idxs)
        prognostic = prediction[..., :num_prognostic]
        new_state[..., self.feature_indices.prognostic_idxs] = prognostic

        # Diagnostic variables remain zero (not carried forward)

        return new_state


class GraphDataHandler(BaseDataHandler):
    """DataHandler for graph data."""

    def prepare_input(self, data: Data) -> Data:
        features = data["grid"].data
        features = self.normalize_features(features)

        input_features = self.get_input_features(features[0])
        residual = self.get_output_features(features[0])

        data["grid"]["input"] = input_features
        data["grid"]["residual"] = residual

        return data

    def get_target(self, data: Data) -> torch.Tensor:
        features = data["grid"].data[1]
        target = self.get_output_features(features)
        target = self.normalize_output_features(target)
        return target

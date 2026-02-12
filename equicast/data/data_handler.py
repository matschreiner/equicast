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

    normalizer: Normalizer

    def __init__(self, dataset_path: str, feature_config: FeatureConfig):
        super().__init__()
        data = open_dataset(dataset_path)
        self.statistics: dict[str, torch.Tensor] = cast_dict(
            data.statistics, torch.Tensor
        )
        self.name_to_index: dict[str, int] = data.name_to_index

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
    def update_state_with_prediction(self, state: Any, pred_state: Any) -> Any: ...

    @abstractmethod
    def prepare_backbone_input(self, data: Any) -> Any: ...

    @abstractmethod
    def update_state_with_backbone_output(
        self, state: Any, backbone_output: torch.Tensor
    ) -> Any: ...

    @abstractmethod
    def prepare_backbone_target(self, data: Any) -> torch.Tensor: ...

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

    def __init__(
        self,
        dataset_path: str,
        feature_config: FeatureConfig,
        nodes: str = "grid",
    ):
        super().__init__(dataset_path, feature_config)
        self.nodes = nodes
        self.normalizer = Normalizer(self.statistics)

    def prepare_backbone_input(self, data: Data) -> Data:
        normalized = self.normalize_features(data[self.nodes].data)
        data[self.nodes]["input"] = self.get_input_features(normalized)
        data[self.nodes]["residual"] = self.get_output_features(normalized)

        return data

    def prepare_backbone_target(self, data: Data) -> torch.Tensor:
        normalized = self.normalize_features(data[self.nodes].data)
        target = self.get_output_features(normalized)
        return target

    def update_state_with_backbone_output(
        self, state: Data, backbone_output: torch.Tensor
    ) -> Data:
        """Denormalize backbone output and write to state's data tensor."""
        output = self.inverse_normalize_output_features(backbone_output)
        state[self.nodes].data[..., self.out_idxs] = output
        return state

    def update_state_with_prediction(self, state: Data, pred_state: Data) -> Data:
        """Copy output features from pred_state to state."""
        pred = pred_state[self.nodes].data[..., self.out_idxs]
        state[self.nodes].data[..., self.out_idxs] = pred
        return state

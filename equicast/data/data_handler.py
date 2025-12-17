"""DataHandler for managing scaling and feature routing metadata."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from anemoi.datasets import open_dataset
from torch_geometric.data import Data

from equicast.data.feature_config import FeatureConfig
from equicast.data.feature_router import FeatureRouter
from equicast.data.scaler import Scaler
from equicast.utils.utils import cast_dict


class BaseDataHandler(torch.nn.Module, ABC):
    """Abstract base class for data handlers."""

    def __init__(self, dataset_path: str, feature_config: FeatureConfig):
        super().__init__()
        self.dataset_path = dataset_path
        self.feature_config = feature_config

        # Open dataset to extract metadata
        data = open_dataset(dataset_path)
        self.statistics: dict[str, torch.Tensor] = cast_dict(
            data.statistics, torch.Tensor
        )
        self.name_to_index: dict[str, int] = data.name_to_index

    @abstractmethod
    def prepare_input(self, data: Any) -> Any:
        """Convert raw state to model input."""
        pass

    @abstractmethod
    def get_target(self, data: Any) -> Any:
        """Convert raw state to model target."""
        pass

    @abstractmethod
    def update_state_with_prediction(
        self, state: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Update state with new prediction."""
        pass

    @abstractmethod
    def pad_input_features(self, input_features: torch.Tensor) -> torch.Tensor:
        """Pad input features to full feature set."""
        pass

    @abstractmethod
    def pad_output_features(self, output_features: torch.Tensor) -> torch.Tensor:
        """Pad output features to full feature set."""
        pass


class DataHandler(BaseDataHandler):
    """Standard DataHandler with z-score normalization."""

    def __init__(self, dataset_path: str, feature_config: FeatureConfig):
        super().__init__(dataset_path, feature_config)
        self.scaler = Scaler(self.statistics)
        self.feature_router = FeatureRouter(
            feature_config=feature_config,
            name_to_index=self.name_to_index,
        )

    @property
    def in_idxs(self) -> list[int]:
        return self.feature_router.in_idxs

    @property
    def out_idxs(self) -> list[int]:
        return self.feature_router.out_idxs

    def update_state_with_prediction(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        new_state = state.clone()
        new_state = self.scaler.transform(new_state)
        new_state[..., self.out_idxs] = prediction
        return self.scaler.inverse_transform(new_state)

    def prepare_input(self, data: Data) -> Data:
        graph = data
        raw = graph["grid"].data[0]
        scaled = self.scaler.transform(raw)
        graph["grid"].input = self.get_input_features(scaled)
        graph["grid"].residual = self.get_output_features(scaled)
        return graph

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

    def get_target(self, data: Data) -> torch.Tensor:
        raw = data["grid"].data[1]
        scaled = self.scaler.transform(raw)
        return self.get_output_features(scaled)

"""DataHandler for managing scaling and feature routing metadata."""

from abc import ABC, abstractmethod

import torch
from anemoi.datasets import open_dataset

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

    @property
    @abstractmethod
    def in_idxs(self) -> list[int]:
        """Input feature indices."""
        pass

    @property
    @abstractmethod
    def out_idxs(self) -> list[int]:
        """Output feature indices."""
        pass

    @abstractmethod
    def prepare_model_input(self, raw_state: torch.Tensor) -> torch.Tensor:
        """Convert raw state to model input."""
        pass

    @abstractmethod
    def prepare_model_target(self, raw_state: torch.Tensor) -> torch.Tensor:
        """Convert raw state to model target."""
        pass

    @abstractmethod
    def update_state_with_prediction(
        self, state: torch.Tensor, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Update state with new prediction."""
        pass


class DataHandler(BaseDataHandler):
    """Standard DataHandler with z-score normalization."""

    def __init__(self, dataset_path: str, feature_config: FeatureConfig):
        super().__init__(dataset_path, feature_config)

        # Initialize scaler and feature router
        self.scaler = Scaler(self.statistics)
        self.feature_router = FeatureRouter(
            feature_config=feature_config,
            name_to_index=self.name_to_index,
        )

    @property
    def in_idxs(self) -> list[int]:
        """Input feature indices (forcing + prognostic)."""
        return self.feature_router.in_idxs

    @property
    def out_idxs(self) -> list[int]:
        """Output feature indices (prognostic + diagnostic)."""
        return self.feature_router.out_idxs

    def prepare_model_input(self, raw_state: torch.Tensor) -> torch.Tensor:
        scaled = self.scaler.transform(raw_state)
        return scaled[..., self.in_idxs]

    def prepare_model_target(self, raw_state: torch.Tensor) -> torch.Tensor:
        scaled = self.scaler.transform(raw_state)
        return scaled[..., self.out_idxs]

    def update_state_with_prediction(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        new_state = state.clone()
        new_state = self.scaler(new_state)
        new_state[..., self.out_idxs] = prediction
        return self.scaler.inverse_transform(new_state)

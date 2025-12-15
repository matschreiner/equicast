"""DataHandler for managing scaling and feature routing metadata."""

import torch
from anemoi.datasets import open_dataset

from equicast.data.feature_config import FeatureConfig
from equicast.data.feature_router import FeatureRouter
from equicast.data.scaler import Scaler
from equicast.utils.utils import cast_dict


class DataHandler:
    """
    Lightweight handler for dataset metadata (statistics, feature mappings).

    Extracts only the metadata needed for scaling and feature routing,
    without loading the full dataset into memory.
    """

    def __init__(self, dataset_path: str, feature_config: FeatureConfig):
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
        self.statistics: dict[str, torch.Tensor] = cast_dict(
            data.statistics, torch.Tensor
        )
        self.name_to_index: dict[str, int] = data.name_to_index

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
        """
        Convert raw state to model input.

        Applies z-score normalization and routes to input features (forcing + prognostic).

        Args:
            raw_state: Raw data [nodes, all_features] in physical space

        Returns:
            Model input [nodes, input_features] in normalized space
        """
        scaled = self.scaler.transform(raw_state)
        return scaled[..., self.in_idxs]

    def prepare_model_target(self, raw_state: torch.Tensor) -> torch.Tensor:
        """
        Convert raw state to model target.

        Applies z-score normalization and routes to output features (prognostic + diagnostic).

        Args:
            raw_state: Raw data [nodes, all_features] in physical space

        Returns:
            Model target [nodes, output_features] in normalized space
        """
        scaled = self.scaler.transform(raw_state)
        return scaled[..., self.out_idxs]

    def update_state_with_prediction(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update state with new prediction.

        Takes current state and model prediction, extracts prognostic from prediction,
        and updates the state. Useful for autoregressive forecasting.

        Args:
            state: Current state [nodes, all_features] in physical space
            prediction: Model prediction [nodes, output_features] in scaled space
            forcing: Optional updated forcing [nodes, n_forcing]. If None, keeps current forcing

        Returns:
            Updated state [nodes, all_features] in physical space

        Examples:
            >>> # Update with prediction, keep same forcing
            >>> new_state = handler.update_state_with_prediction(state, model_pred)

            >>> # Update with prediction and new forcing
            >>> new_state = handler.update_state_with_prediction(state, model_pred, new_forcing)
        """

        new_state = state.clone()
        new_state = self.scaler(new_state)
        new_state[..., self.out_idxs] = prediction
        return self.scaler.inverse_transform(new_state)

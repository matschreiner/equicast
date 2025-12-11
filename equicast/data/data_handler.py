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
        scaled = self.scaler(raw_state)
        return scaled[:, self.in_idxs]

    def prepare_model_target(self, raw_state: torch.Tensor) -> torch.Tensor:
        """
        Convert raw state to model target.

        Applies z-score normalization and routes to output features (prognostic + diagnostic).

        Args:
            raw_state: Raw data [nodes, all_features] in physical space

        Returns:
            Model target [nodes, output_features] in normalized space
        """
        scaled = self.scaler(raw_state)
        return scaled[:, self.out_idxs]

    def from_model_output(
        self, model_output: torch.Tensor, forcing: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Convert model output back to raw state.

        Unscales model output, extracts prognostic variables, and reconstructs
        full feature vector by combining with forcing variables.

        Args:
            model_output: Model predictions [nodes, output_features] in normalized space
            forcing: Forcing variables [nodes, n_forcing] in physical space (optional)

        Returns:
            Full state [nodes, all_features] in physical space
        """
        prognostic = self.extract_prognostic(model_output)
        return self.reconstruct_state(prognostic, forcing)

    def extract_prognostic(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Extract prognostic variables from model prediction.

        Prediction contains [prognostic, diagnostic] variables in scaled space.
        This method unscales and extracts only the prognostic variables.

        Args:
            prediction: Model output [prognostic, diagnostic] in scaled space

        Returns:
            Prognostic variables only, in physical space
        """
        # Get indices for output features (prognostic + diagnostic)
        out_idxs = self.out_idxs

        # Get mean/std for output features only
        mean_out = self.scaler.mean[out_idxs]
        std_out = self.scaler.std[out_idxs]

        # Unscale prediction to physical space
        pred_unscaled = prediction * std_out + mean_out

        # Extract prognostic (first N features)
        n_prognostic = len(self.feature_router.feature_config.prognostic)
        prognostic = pred_unscaled[:, :n_prognostic]

        return prognostic

    def reconstruct_state(
        self, prognostic: torch.Tensor, forcing: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Reconstruct full feature vector from prognostic and forcing variables.

        Creates a full feature vector with all features in their correct positions.
        Non-provided features (e.g., diagnostic) are left as zeros.

        Args:
            prognostic: Prognostic variables in physical space
            forcing: Forcing variables in physical space (optional)

        Returns:
            Full feature vector [n_nodes, n_features] in physical space
        """
        n_features = len(self.name_to_index)
        state = torch.zeros(prognostic.shape[0], n_features, device=prognostic.device)

        # Place prognostic variables at their correct indices
        prog_idxs = self.feature_router._get_data_idxs(
            self.feature_router.feature_config.prognostic
        )
        for i, idx in enumerate(prog_idxs):
            state[:, idx] = prognostic[:, i]

        # Place forcing variables at their correct indices
        if forcing is not None:
            forcing_idxs = self.feature_router._get_data_idxs(
                self.feature_router.feature_config.forcing
            )
            for i, idx in enumerate(forcing_idxs):
                state[:, idx] = forcing[:, i]

        return state

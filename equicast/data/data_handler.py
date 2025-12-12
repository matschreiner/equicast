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
        self,
        *,
        prognostic: torch.Tensor = None,
        forcing: torch.Tensor = None,
        diagnostic: torch.Tensor = None,
        prediction: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Reconstruct full feature vector from any combination of components.

        Uses keyword-only arguments for flexibility - provide any combination of
        prognostic, forcing, diagnostic, or prediction (which will be processed
        to extract prognostic automatically).

        Args:
            prognostic: Prognostic variables in physical space [nodes, n_prog]
            forcing: Forcing variables in physical space [nodes, n_forcing]
            diagnostic: Diagnostic variables in physical space [nodes, n_diag]
            prediction: Model prediction in scaled space (will extract prognostic automatically)

        Returns:
            Full state [nodes, all_features] in physical space

        Examples:
            >>> # From model prediction + forcing
            >>> state = reconstruct_state(prediction=pred, forcing=forcing)

            >>> # From prognostic only
            >>> state = reconstruct_state(prognostic=prog_vars)

            >>> # From prognostic + forcing + diagnostic
            >>> state = reconstruct_state(prognostic=p, forcing=f, diagnostic=d)
        """
        n_features = len(self.name_to_index)

        # Determine device from any provided tensor
        device = None
        for tensor in [prognostic, forcing, diagnostic, prediction]:
            if tensor is not None:
                device = tensor.device
                break
        if device is None:
            device = torch.device("cpu")

        # Get number of nodes from any provided tensor
        n_nodes = None
        for tensor in [prognostic, forcing, diagnostic, prediction]:
            if tensor is not None:
                n_nodes = tensor.shape[0]
                break

        if n_nodes is None:
            raise ValueError("At least one argument must be provided")

        state = torch.zeros(n_nodes, n_features, device=device)

        # If prediction provided, extract prognostic from it
        if prediction is not None:
            prognostic = self.extract_prognostic(prediction)

        # Place each component at its correct indices
        if prognostic is not None:
            prog_idxs = self.feature_router._get_data_idxs(
                self.feature_router.feature_config.prognostic
            )
            for i, idx in enumerate(prog_idxs):
                state[:, idx] = prognostic[:, i]

        if forcing is not None:
            forcing_idxs = self.feature_router._get_data_idxs(
                self.feature_router.feature_config.forcing
            )
            for i, idx in enumerate(forcing_idxs):
                state[:, idx] = forcing[:, i]

        if diagnostic is not None:
            diag_idxs = self.feature_router._get_data_idxs(
                self.feature_router.feature_config.diagnostic
            )
            for i, idx in enumerate(diag_idxs):
                state[:, idx] = diagnostic[:, i]

        return state

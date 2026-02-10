"""DataHandler for equivariant models with separate scalar and vector features."""

from typing import Any

import torch
from anemoi.datasets import open_dataset
from torch_geometric.data import Data

from equicast.data.data_handler import BaseDataHandler
from equicast.data.feature_config import FeatureConfig
from equicast.data.normalizer import VectorNormalizer


class EquivariantGraphDataHandler(BaseDataHandler):
    """DataHandler that packs vector features (e.g., wind) into [n, num_vectors, 2] tensors."""

    def __init__(
        self,
        dataset_path: str,
        feature_config: FeatureConfig,
        nodes: str = "grid",
    ):
        super().__init__(dataset_path, feature_config)
        self.nodes = nodes
        self.feature_config = feature_config

        # Build vector index pairs (u_idx, v_idx) for each vector
        self.prognostic_vector_idxs = self._get_vector_idxs(
            feature_config.prognostic_vector
        )

        # Create normalizer with vector support
        data = open_dataset(dataset_path)
        self.normalizer = VectorNormalizer(
            self.statistics,
            data,
            self.prognostic_vector_idxs,
        )

    def _get_vector_idxs(
        self, vector_config: dict[str, list[str]]
    ) -> list[tuple[int, int]]:
        """Get (u_idx, v_idx) pairs for each vector feature."""
        return [
            (
                self.name_to_index[components[0]],
                self.name_to_index[components[1]],
            )
            for components in vector_config.values()
        ]

    def _pack_vectors(self, data: torch.Tensor) -> torch.Tensor:
        """Pack vector components into [nodes, num_vectors, 2] tensor."""
        if not self.prognostic_vector_idxs:
            return torch.empty(*data.shape[:-1], 0, 2, device=data.device)

        u_idxs = [pair[0] for pair in self.prognostic_vector_idxs]
        v_idxs = [pair[1] for pair in self.prognostic_vector_idxs]

        u_components = data[..., u_idxs]  # [nodes, num_vectors]
        v_components = data[..., v_idxs]  # [nodes, num_vectors]

        # Stack to [nodes, num_vectors, 2]
        return torch.stack([u_components, v_components], dim=-1)

    def _unpack_vectors(
        self, vectors: torch.Tensor, data: torch.Tensor
    ) -> torch.Tensor:
        """Unpack [nodes, num_vectors, 2] back into data tensor."""
        if not self.prognostic_vector_idxs:
            return data

        u_idxs = [pair[0] for pair in self.prognostic_vector_idxs]
        v_idxs = [pair[1] for pair in self.prognostic_vector_idxs]

        data[..., u_idxs] = vectors[..., 0]
        data[..., v_idxs] = vectors[..., 1]
        return data

    def normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Normalize packed vectors."""
        return self.normalizer.transform_vectors(vectors)  # type: ignore

    def inverse_normalize_vectors(self, vectors: torch.Tensor) -> torch.Tensor:
        """Inverse normalize packed vectors."""
        return self.normalizer.inverse_transform_vectors(vectors)  # type: ignore

    def prepare_input(self, data: Data) -> Data:
        raw = data[self.nodes].data

        # Normalize scalars
        normalized_scalars = self.normalize_features(raw)
        data[self.nodes]["input_scalar"] = self.get_input_features(
            normalized_scalars
        )
        data[self.nodes]["residual_scalar"] = self.get_output_features(
            normalized_scalars
        )

        # Pack and normalize vectors
        vectors = self._pack_vectors(raw)
        normalized_vectors = self.normalize_vectors(vectors)
        data[self.nodes]["input_vector"] = normalized_vectors
        data[self.nodes]["residual_vector"] = normalized_vectors

        return data

    def prepare_backbone_target(self, data: Data) -> dict[str, torch.Tensor]:
        """Prepare normalized scalar and vector targets."""
        raw = data[self.nodes].data

        # Scalar target
        normalized = self.normalize_features(raw)
        scalar_target = self.get_output_features(normalized)

        # Vector target
        vectors = self._pack_vectors(raw)
        vector_target = self.normalize_vectors(vectors)

        return {"scalar": scalar_target, "vector": vector_target}

    def update_state_with_backbone_output(
        self,
        state: Data,
        backbone_output: Any,
    ) -> Data:
        """Denormalize outputs and write to state's data tensor."""
        # Denormalize scalars
        scalars = self.inverse_normalize_output_features(
            backbone_output["scalar"]
        )
        state[self.nodes].data[..., self.out_idxs] = scalars

        # Denormalize and unpack vectors
        vectors = self.inverse_normalize_vectors(backbone_output["vector"])
        self._unpack_vectors(vectors, state[self.nodes].data)

        return state

    def update_state_with_prediction(
        self, state: Data, pred_state: Data
    ) -> Data:
        """Copy output features from pred_state to state."""
        pred = pred_state[self.nodes].data[..., self.out_idxs]
        state[self.nodes].data[..., self.out_idxs] = pred
        return state

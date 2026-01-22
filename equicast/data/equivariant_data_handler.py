"""DataHandler for equivariant models with separate scalar and vector features."""

import torch
from torch_geometric.data import Data

from equicast.data.data_handler import BaseDataHandler
from equicast.data.feature_config import FeatureConfig


class EquivariantDataHandler(BaseDataHandler):
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

    def _get_vector_idxs(
        self, vector_config: dict[str, list[str]]
    ) -> list[tuple[int, int]]:
        """Get (u_idx, v_idx) pairs for each vector feature."""
        return [
            (self.name_to_index[components[0]], self.name_to_index[components[1]])
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

    def prepare_input(self, data: Data) -> Data:
        normalized = self.normalize_features(data[self.nodes].data)

        # Scalar features
        data[self.nodes]["input_scalar"] = self.get_input_features(normalized)
        data[self.nodes]["residual_scalar"] = self.get_output_features(normalized)

        # Vector features [nodes, num_vectors, 2]
        data[self.nodes]["input_vector"] = self._pack_vectors(normalized)
        data[self.nodes]["residual_vector"] = self._pack_vectors(normalized)

        return data

    def prepare_backbone_target(self, data: Data) -> torch.Tensor:
        normalized = self.normalize_features(data[self.nodes].data)
        target = self.get_output_features(normalized)
        return target

    def update_output(self, input: Data, backbone_out: torch.Tensor) -> Data:
        output = self.inverse_normalize_output_features(backbone_out)
        input[self.nodes].data[..., self.out_idxs] = output
        return input

    def update_next_with_prediction(
        self, target_graph: Data, pred_graph: Data
    ) -> Data:
        """Create next input graph from prediction + forcing."""
        pred = pred_graph[self.nodes].data[..., self.out_idxs]
        target_graph[self.nodes].data[..., self.out_idxs] = pred
        return target_graph

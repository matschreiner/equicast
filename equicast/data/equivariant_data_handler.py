"""DataHandler for equivariant models with separate scalar and vector features."""

from typing import Any

import numpy as np
import torch
from anemoi.datasets import open_dataset
from torch_geometric.data import Data

from equicast.data.data_handler import BaseDataHandler
from equicast.data.feature_config import FeatureConfig
from equicast.data.normalizer import Normalizer, VectorNormalizer


def compute_vector_mean_norm(
    dataset,
    vector_indices: list[tuple[int, int]],
    num_samples: int = 100,
) -> torch.Tensor:
    if not vector_indices:
        return torch.tensor([])

    num_samples = min(num_samples, len(dataset))
    sample_indices = np.random.choice(
        len(dataset),
        num_samples,
        replace=False,
    )
    samples = dataset[sample_indices].squeeze()  # [samples, features, nodes]
    samples = samples.transpose(0, 2, 1)  # [samples, nodes, features]

    u_idxs, v_idxs = zip(*vector_indices)
    u = samples[..., u_idxs]
    v = samples[..., v_idxs]
    norms = np.sqrt(u**2 + v**2)

    mean_norms = norms.mean(axis=tuple(range(norms.ndim - 1)))
    return torch.tensor(mean_norms)


class EquivariantGraphDataHandler(BaseDataHandler):
    """DataHandler that packs vector features (e.g., wind) into [n, num_vectors, 2] tensors."""

    @property
    def in_vector_dim(self) -> int:
        return len(self.in_vector_idxs)

    @property
    def out_vector_dim(self) -> int:
        return len(self.out_vector_idxs)

    def __init__(
        self,
        dataset_path: str,
        feature_config: FeatureConfig,
        nodes: str = "grid",
    ):
        super().__init__(dataset_path, feature_config)
        self.nodes = nodes

        # Build vector index pairs (u_idx, v_idx) for input/output
        forcing_vector_idxs = self._get_vector_idxs(feature_config.forcing_vector)
        prognostic_vector_idxs = self._get_vector_idxs(feature_config.prognostic_vector)
        diagnostic_vector_idxs = self._get_vector_idxs(feature_config.diagnostic_vector)
        self.in_vector_idxs = forcing_vector_idxs + prognostic_vector_idxs
        self.out_vector_idxs = prognostic_vector_idxs + diagnostic_vector_idxs
        if self.in_vector_idxs == self.out_vector_idxs:
            self.get_output_vectors = self.get_input_vectors

        # Create normalizers
        data = open_dataset(dataset_path)
        vector_mean_norm = compute_vector_mean_norm(data, self.out_vector_idxs)

        self.normalizer = Normalizer(self.statistics, self.in_idxs, self.out_idxs)
        self.vector_normalizer = VectorNormalizer(vector_mean_norm)

    def prepare_backbone_input(self, data: Data) -> Data:
        raw = data[self.nodes].data

        # Select then normalize scalars
        input_scalars = self.get_input_scalars(raw)
        output_scalars = self.get_output_scalars(raw)
        data[self.nodes]["input_scalar"] = self.normalizer.normalize_input(input_scalars)
        data[self.nodes]["residual_scalar"] = self.normalizer.normalize_output(output_scalars)

        # Pack and normalize vectors
        input_vectors = self.get_input_vectors(raw)
        norm_input_vectors = self.vector_normalizer.normalize_vectors(input_vectors)
        data[self.nodes]["input_vector"] = norm_input_vectors

        if self._shared_vector_idxs:
            data[self.nodes]["residual_vector"] = norm_input_vectors
        else:
            output_vectors = self.get_output_vectors(raw)
            data[self.nodes]["residual_vector"] = self.vector_normalizer.normalize_vectors(output_vectors)

        return data

    def prepare_backbone_target(self, data: Data) -> dict[str, torch.Tensor]:
        raw = data[self.nodes].data

        output_scalars = self.get_output_scalars(raw)
        scalar_target = self.normalizer.normalize_output(output_scalars)

        output_vectors = self.get_output_vectors(raw)
        vector_target = self.vector_normalizer.normalize_vectors(output_vectors)

        return {"scalar": scalar_target, "vector": vector_target}

    def update_state_with_backbone_output(
        self,
        state: Data,
        backbone_output: Any,
    ) -> Data:
        # Denormalize scalars
        denormalized = self.normalizer.denormalize_output(backbone_output["scalar"])
        self.set_output_scalars(state[self.nodes].data, denormalized)

        # Denormalize and unpack vectors
        vectors = self.vector_normalizer.denormalize_vectors(backbone_output["vector"])
        self.set_output_vectors(state[self.nodes].data, vectors)

        return state

    def update_state_with_prediction(self, state: Data, pred_state: Data) -> Data:
        """Copy output features from pred_state to state."""
        pred = self.get_output_scalars(pred_state[self.nodes].data)
        self.set_output_scalars(state[self.nodes].data, pred)
        return state

    def _get_vector_idxs(self, vector_config: dict[str, list[str]]) -> list[tuple[int, int]]:
        """Get (u_idx, v_idx) pairs for each vector feature."""
        return [
            (
                self.name_to_index[components[0]],
                self.name_to_index[components[1]],
            )
            for components in vector_config.values()
        ]

    def _pack_vectors(self, raw: torch.Tensor, vector_idxs: list[tuple[int, int]]) -> torch.Tensor:
        """Pack vector components into [..., num_vectors, 2] tensor."""
        if not vector_idxs:
            return torch.empty(*raw.shape[:-1], 0, 2, device=raw.device)

        u_idxs = [pair[0] for pair in vector_idxs]
        v_idxs = [pair[1] for pair in vector_idxs]

        u_components = raw[..., u_idxs]
        v_components = raw[..., v_idxs]

        return torch.stack([u_components, v_components], dim=-1)

    def get_input_vectors(self, raw: torch.Tensor) -> torch.Tensor:
        return self._pack_vectors(raw, self.in_vector_idxs)

    def get_output_vectors(self, raw: torch.Tensor) -> torch.Tensor:
        return self._pack_vectors(raw, self.out_vector_idxs)

    def set_output_vectors(self, raw: torch.Tensor, vectors: torch.Tensor) -> None:
        if not self.out_vector_idxs:
            return

        u_idxs = [pair[0] for pair in self.out_vector_idxs]
        v_idxs = [pair[1] for pair in self.out_vector_idxs]

        raw[..., u_idxs] = vectors[..., 0]
        raw[..., v_idxs] = vectors[..., 1]

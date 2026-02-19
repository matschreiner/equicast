"""DataHandler for equivariant models with separate scalar and vector features."""

import os
from typing import Any

import numpy as np
import torch
import xarray as xr
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

    def __init__(
        self,
        dataset_path: str,
        feature_config: FeatureConfig,
        nodes: str = "grid",
    ):
        super().__init__(dataset_path, feature_config)
        self.nodes = nodes
        self._vectors_in_eq_out = (
            self.feature_index.in_vector_idxs == self.feature_index.out_vector_idxs
        )

        # Create normalizers
        data = open_dataset(dataset_path)
        vector_mean_norm = compute_vector_mean_norm(data, self.feature_index.out_vector_idxs)

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
        norm_input_vectors = self.vector_normalizer.normalize_vectors(self.get_input_vectors(raw))
        norm_output_vectors = (
            self.vector_normalizer.normalize_vectors(self.get_output_vectors(raw))
            if not self._vectors_in_eq_out
            else norm_input_vectors
        )
        data[self.nodes]["input_vector"] = norm_input_vectors
        data[self.nodes]["residual_vector"] = norm_output_vectors

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

    def to_cf(self, graph: Data) -> "xr.Dataset":
        """Convert a graph state to a CF-compliant xarray Dataset."""
        import xarray as xr

        index_to_name = {v: k for k, v in self.name_to_index.items()}

        # Collect all output indices: scalars + vector components
        vector_idxs = []
        for u_idx, v_idx in self.feature_index.out_vector_idxs:
            vector_idxs.extend([u_idx, v_idx])
        all_idxs = self.out_idxs + vector_idxs

        data = graph[self.nodes].data[..., all_idxs].cpu().numpy()

        latlon = graph[self.nodes].x.cpu().numpy()
        lat = np.degrees(latlon[:, 0])
        lon = np.degrees(latlon[:, 1])

        feature_names = [index_to_name.get(idx, f"feature_{i}") for i, idx in enumerate(all_idxs)]
        ds = xr.Dataset(
            {name: ("node", data[..., i]) for i, name in enumerate(feature_names)},
            coords={
                "latitude": ("node", lat),
                "longitude": ("node", lon),
            },
        )
        ds["latitude"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
        ds["longitude"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
        ds.attrs["Conventions"] = "CF-1.8"
        return ds

    def outputs_to_zarr(self, graphs: list[Data], path: str):
        datasets = [self.to_cf(graph) for graph in graphs]
        combined = xr.concat(datasets, dim="time")
        combined.to_zarr(path, mode="w")

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
        return self._pack_vectors(raw, self.feature_index.in_vector_idxs)

    def get_output_vectors(self, raw: torch.Tensor) -> torch.Tensor:
        return self._pack_vectors(raw, self.feature_index.out_vector_idxs)

    def set_output_vectors(self, raw: torch.Tensor, vectors: torch.Tensor) -> None:
        if not self.feature_index.out_vector_idxs:
            return

        u_idxs = [pair[0] for pair in self.feature_index.out_vector_idxs]
        v_idxs = [pair[1] for pair in self.feature_index.out_vector_idxs]

        raw[..., u_idxs] = vectors[..., 0]
        raw[..., v_idxs] = vectors[..., 1]

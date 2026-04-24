"""DataHandler for equivariant models with separate scalar and vector features."""

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
        self._vectors_in_eq_out = self.feature_index.in_vector_idxs == self.feature_index.out_vector_idxs

        # Create normalizers
        data = open_dataset(dataset_path)
        vector_mean_norm = compute_vector_mean_norm(data, self.feature_index.out_vector_idxs)

        self.normalizer = Normalizer(self.statistics, self.in_idxs, self.out_idxs)
        self.vector_normalizer = VectorNormalizer(vector_mean_norm)

    def prepare_training_batch(self, batch: list[Data]) -> tuple:
        backbone_input = self.prepare_backbone_input(batch[0])  # type: ignore
        backbone_target = self.prepare_backbone_target(batch[1])  # type: ignore
        return backbone_input, backbone_target

    def prepare_backbone_frame(self, phys_input: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = phys_input[self.nodes].data
        scalars = self.normalizer.normalize_input(self.get_input_scalars(raw))
        vectors = self.vector_normalizer.normalize_vectors(self.get_input_vectors(raw))
        output_scalars = self.get_output_scalars(raw)
        residual_scalar = self.normalizer.normalize_output(output_scalars)
        residual_vector = (
            self.vector_normalizer.normalize_vectors(self.get_output_vectors(raw))
            if not self._vectors_in_eq_out
            else vectors
        )

        return scalars, vectors, residual_scalar, residual_vector

    def prepare_backbone_input(self, phys_input: Data) -> Data:
        scalars, vectors, residual_scalar, residual_vector = self.prepare_backbone_frame(phys_input)
        phys_input[self.nodes]["input_scalar"] = scalars
        phys_input[self.nodes]["input_vector"] = vectors
        phys_input[self.nodes]["residual_scalar"] = residual_scalar
        phys_input[self.nodes]["residual_vector"] = residual_vector
        return phys_input

    def prepare_backbone_target(self, phys_input: Data) -> dict[str, torch.Tensor]:
        raw = phys_input[self.nodes].data

        output_scalars = self.get_output_scalars(raw)
        scalar_target = self.normalizer.normalize_output(output_scalars)

        output_vectors = self.get_output_vectors(raw)
        vector_target = self.vector_normalizer.normalize_vectors(output_vectors)

        return {"scalar": scalar_target, "vector": vector_target}

    def backbone_out_to_phys_out(self, backbone_output: dict) -> torch.Tensor:
        scalar = self.normalizer.denormalize_output(backbone_output["scalar"])
        vectors = self.vector_normalizer.denormalize_vectors(backbone_output["vector"])
        raw = torch.zeros(scalar.shape[0], self.num_features, device=scalar.device, dtype=scalar.dtype)
        self.set_output_scalars(raw, scalar)
        self.set_output_vectors(raw, vectors)
        return raw

    def update_state_with_backbone_output(
        self,
        phys_input: Data,
        backbone_output: Any,
    ) -> Data:
        if isinstance(backbone_output, torch.Tensor):
            denormalized = self.normalizer.denormalize_output(backbone_output)
            self.set_output_scalars(phys_input[self.nodes].data, denormalized)
            return phys_input

        denormalized = self.normalizer.denormalize_output(backbone_output["scalar"])
        self.set_output_scalars(phys_input[self.nodes].data, denormalized)

        vectors = self.vector_normalizer.denormalize_vectors(backbone_output["vector"])
        self.set_output_vectors(phys_input[self.nodes].data, vectors)

        return phys_input

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

    @property
    def in_vector_dim(self) -> int:
        return self.feature_index.in_vector_dim

    @property
    def out_vector_dim(self) -> int:
        return self.feature_index.out_vector_dim

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


class MultiFrameEquivariantGraphDataHandler(EquivariantGraphDataHandler):
    """Extension of EquivariantGraphDataHandler that supports multiframe inputs/outputs."""

    def __init__(
        self,
        dataset_path: str,
        feature_config: FeatureConfig,
        nodes: str = "grid",
        n_input_frames=2,
    ):
        self.n_input_frames = n_input_frames
        super().__init__(dataset_path, feature_config, nodes)
        self.n_input_frames = n_input_frames

    @property
    def in_dim(self) -> int:
        return self.n_input_frames * self.feature_index.in_dim

    @property
    def in_vector_dim(self) -> int:
        return self.n_input_frames * self.feature_index.in_vector_dim

    def prepare_training_batch(self, batch: list[Data]) -> tuple:
        input_ = self.prepare_backbone_input(batch[: self.n_input_frames])  # type: ignore
        target = self.prepare_backbone_target(batch[self.n_input_frames])  # type: ignore
        return input_, target

    def prepare_backbone_input(self, frames: list[Data]) -> Data:  # type: ignore
        frame_data = [self.prepare_backbone_frame(f) for f in frames]
        all_scalars = [d[0] for d in frame_data]
        all_vectors = [d[1] for d in frame_data]
        _, _, residual_scalar, residual_vector = frame_data[-1]
        data = frames[-1]
        data[self.nodes]["input_scalar"] = torch.cat(all_scalars, dim=-1)
        data[self.nodes]["input_vector"] = torch.cat(all_vectors, dim=-2)
        data[self.nodes]["residual_scalar"] = residual_scalar
        data[self.nodes]["residual_vector"] = residual_vector
        return data

"""DataHandler for managing scaling and feature routing metadata."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from anemoi.datasets import open_dataset
from torch_geometric.data import Data

from equicast.data.feature_config import FeatureConfig
from equicast.data.feature_index import FeatureIndex
from equicast.data.normalizer import Normalizer
from equicast.utils.utils import cast_dict


class BaseDataHandler(torch.nn.Module, ABC):
    """Standard DataHandler with z-score normalization."""

    normalizer: Normalizer

    def __init__(self, dataset_path: str, feature_config: FeatureConfig):
        super().__init__()
        self.feature_config = feature_config
        data = open_dataset(dataset_path)
        self.statistics: dict[str, torch.Tensor] = cast_dict(data.statistics, torch.Tensor)
        self.name_to_index: dict[str, int] = data.name_to_index
        self.num_features: int = max(self.name_to_index.values()) + 1

        self.feature_index = FeatureIndex(
            feature_config=feature_config,
            name_to_index=self.name_to_index,
        )

    @property
    def in_idxs(self) -> list[int]:
        return self.feature_index.in_idxs

    @property
    def out_idxs(self) -> list[int]:
        return self.feature_index.out_idxs

    @property
    def in_dim(self) -> int:
        return self.feature_index.in_dim

    @property
    def out_dim(self) -> int:
        return self.feature_index.out_dim

    def get_input_scalars(self, raw: torch.Tensor) -> torch.Tensor:
        return raw[..., self.in_idxs]

    def get_output_scalars(self, raw: torch.Tensor) -> torch.Tensor:
        return raw[..., self.out_idxs]

    def set_output_scalars(self, raw: torch.Tensor, values: torch.Tensor) -> None:
        raw[..., self.out_idxs] = values

    @abstractmethod
    def prepare_training_batch(self, batch: list) -> tuple: ...

    @abstractmethod
    def prepare_backbone_input(self, data: Any) -> Any: ...

    @abstractmethod
    def update_state_with_backbone_output(self, state: Any, backbone_output: Any) -> Any: ...

    @abstractmethod
    def prepare_backbone_target(self, data: Any) -> Any: ...

    @abstractmethod
    def backbone_out_to_phys_out(self, backbone_output: Any) -> torch.Tensor:
        """Denormalize backbone output into a full feature tensor [nodes, num_features],
        indexed by name_to_index."""
        ...

    def to_cf(self, graph: Any):
        raise NotImplementedError

    def outputs_to_zarr(self, graphs: list[Any], path: str):
        raise NotImplementedError


class GraphDataHandler(BaseDataHandler):
    """DataHandler for graph data."""

    def prepare_training_batch(self, batch: list) -> tuple:
        backbone_input = self.prepare_backbone_input(batch[0])
        backbone_target = self.prepare_backbone_target(batch[1])
        return backbone_input, backbone_target

    def __init__(
        self,
        dataset_path: str,
        feature_config: FeatureConfig,
        nodes: str = "grid",
    ):
        super().__init__(dataset_path, feature_config)
        self.nodes = nodes
        self.normalizer = Normalizer(self.statistics, self.in_idxs, self.out_idxs)

    def prepare_backbone_input(self, phys_input: Data) -> Data:
        raw = phys_input[self.nodes].data
        input_scalars = self.get_input_scalars(raw)
        output_scalars = self.get_output_scalars(raw)
        phys_input[self.nodes]["input"] = self.normalizer.normalize_input(input_scalars)
        phys_input[self.nodes]["residual"] = self.normalizer.normalize_output(output_scalars)
        return phys_input

    def prepare_backbone_target(self, phys_input: Data) -> torch.Tensor:
        raw = phys_input[self.nodes].data
        output_scalars = self.get_output_scalars(raw)
        return self.normalizer.normalize_output(output_scalars)

    def update_state_with_backbone_output(self, phys_input: Data, backbone_output: torch.Tensor) -> Data:
        denormalized = self.normalizer.denormalize_output(backbone_output)
        self.set_output_scalars(phys_input[self.nodes].data, denormalized)
        return phys_input

    def backbone_out_to_phys_out(self, backbone_output: torch.Tensor) -> torch.Tensor:
        scalar = self.normalizer.denormalize_output(backbone_output)
        raw = torch.zeros(scalar.shape[0], self.num_features, device=scalar.device, dtype=scalar.dtype)
        self.set_output_scalars(raw, scalar)
        return raw

    def to_cf(self, graph: Data) -> "xr.Dataset":
        """Convert a graph state to a CF-compliant xarray Dataset."""
        import xarray as xr

        index_to_name = {v: k for k, v in self.name_to_index.items()}

        data = graph[self.nodes].data[..., self.out_idxs].cpu().numpy()

        latlon = graph[self.nodes].x.cpu().numpy()
        lat = np.degrees(latlon[:, 0])
        lon = np.degrees(latlon[:, 1])

        feature_names = [index_to_name.get(idx, f"feature_{i}") for i, idx in enumerate(self.out_idxs)]
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

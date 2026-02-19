import torch

from equicast.data.feature_config import FeatureConfig


class FeatureIndices:
    """Holds feature indices for model input/output."""

    def __init__(self, feature_config: FeatureConfig, name_to_index: dict[str, int]):
        self.feature_config = feature_config
        self.name_to_index = name_to_index

        self.forcing_idxs = self._get_data_idxs(feature_config.forcing)
        self.prognostic_idxs = self._get_data_idxs(feature_config.prognostic)
        self.diagnostic_idxs = self._get_data_idxs(feature_config.diagnostic)

        self.in_idxs: list[int] = self.forcing_idxs + self.prognostic_idxs
        self.out_idxs: list[int] = self.prognostic_idxs + self.diagnostic_idxs

        self.in_vector_idxs: list[tuple[int, int]] = _get_vector_idxs(
            feature_config.forcing_vector, name_to_index
        ) + _get_vector_idxs(feature_config.prognostic_vector, name_to_index)
        self.out_vector_idxs: list[tuple[int, int]] = _get_vector_idxs(
            feature_config.prognostic_vector, name_to_index
        ) + _get_vector_idxs(feature_config.diagnostic_vector, name_to_index)

    def _get_data_idxs(self, names: list[str]) -> list[int]:
        return [self.name_to_index[name] for name in names]

    @property
    def in_dim(self) -> int:
        return len(self.in_idxs)

    @property
    def out_dim(self) -> int:
        return len(self.out_idxs)

    @property
    def in_vector_dim(self) -> int:
        return len(self.in_vector_idxs)

    @property
    def out_vector_dim(self) -> int:
        return len(self.out_vector_idxs)

    @classmethod
    def from_dataset(cls, dataset_path: str, feature_config: FeatureConfig) -> "FeatureIndices":
        from anemoi.datasets import open_dataset

        return cls(feature_config, open_dataset(dataset_path).name_to_index)


def _get_vector_idxs(vector_config: dict[str, list[str]], name_to_index: dict[str, int]) -> list[tuple[int, int]]:
    return [(name_to_index[c[0]], name_to_index[c[1]]) for c in vector_config.values()]

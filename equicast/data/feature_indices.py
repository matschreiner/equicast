import torch

from equicast.data.feature_config import FeatureConfig


class FeatureIndices:
    """Holds feature indices for model input/output."""

    def __init__(
        self, feature_config: FeatureConfig, name_to_index: dict[str, int]
    ):
        self.feature_config = feature_config
        self.name_to_index = name_to_index

        forcing_idxs = self._get_data_idxs(feature_config.forcing)
        prognostic_idxs = self._get_data_idxs(feature_config.prognostic)
        diagnostic_idxs = self._get_data_idxs(feature_config.diagnostic)

        self.in_idxs: list[int] = forcing_idxs + prognostic_idxs
        self.out_idxs: list[int] = prognostic_idxs + diagnostic_idxs

    def _get_data_idxs(self, names: list[str]) -> list[int]:
        return [self.name_to_index[name] for name in names]

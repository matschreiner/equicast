import torch
from anemoi.utils.config import DotDict


class FeatureRouter:
    def __init__(self, feature_config, name_to_index):
        self.feature_config = feature_config
        self.name_to_index = name_to_index

        forcing_idxs = self._get_data_idxs(feature_config.forcing)
        prognostic_idxs = self._get_data_idxs(feature_config.prognostic)
        diagnostic_idxs = self._get_data_idxs(feature_config.diagnostic)

        self.in_idxs = forcing_idxs + prognostic_idxs
        self.out_idxs = prognostic_idxs + diagnostic_idxs

    def _get_data_idxs(self, names):
        return [self.name_to_index[name] for name in names]

    def __call__(self, data):
        return self.transform(data)

    def transform(self, graph):
        cond = graph["data"].raw[:, 0, self.in_idxs]
        target = graph["data"].raw[:, 1, self.out_idxs]

        graph["data"].cond = cond
        graph["data"].target = target

        return graph

import torch
from anemoi.utils.config import DotDict


class FeatureRouter:
    def __init__(self, features, name_to_index):
        self.features = features
        self.name_to_index = name_to_index

        forcing_idxs = self._get_data_idxs(features["forcing"])
        prognostic_idxs = self._get_data_idxs(features["prognostic"])
        diagnostic_idxs = self._get_data_idxs(features["diagnostic"])

        self.input_idxs = forcing_idxs + prognostic_idxs
        self.output_idxs = prognostic_idxs + diagnostic_idxs

    def _get_data_idxs(self, names):
        return [self.name_to_index[name] for name in names]

    def __call__(self, data):
        return self.transform(data)

    def transform(self, graph):
        input = graph.cond[:, self.input_idxs]
        output = graph.target[:, self.output_idxs]
        graph.input = input
        graph.output = output
        return graph

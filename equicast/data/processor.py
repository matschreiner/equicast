from abc import ABC, abstractmethod


class DataProcessor(ABC):
    @abstractmethod
    def preprocess(self, data): ...

    @abstractmethod
    def postprocess(self, data): ...


class SimpleDataProcessor(DataProcessor):
    @property
    def in_dim(self):
        return len(self.in_idxs)

    @property
    def out_dim(self):
        return len(self.out_idxs)

    def __init__(self, statistics, name_to_index, features):
        self.statistics = statistics
        self.name_to_index = name_to_index
        self.features = features

        forcing_idxs = self._get_idxs(features["forcing"])
        prognostic_idxs = self._get_idxs(features["prognostic"])
        diagnostic_idxs = self._get_idxs(features["diagnostic"])

        self.in_idxs = forcing_idxs + prognostic_idxs
        self.out_idxs = prognostic_idxs + diagnostic_idxs

    def _get_idxs(self, names):
        return [self.name_to_index[name] for name in names]

    def route_features(self, data):
        cond = data.raw[:, 0, self.in_idxs]
        target = data.raw[:, 1, self.out_idxs]
        data.cond = cond
        data.target = target
        return data

    def scale_data(self, data):
        mean = self.statistics["mean"]
        std = self.statistics["stdev"]

        mean_in = mean[..., self.in_idxs]
        std_in = std[..., self.in_idxs]
        mean_out = mean[..., self.out_idxs]
        std_out = std[..., self.out_idxs]

        data.cond = (data.cond - mean_in) / std_in
        data.target = (data.target - mean_out) / std_out
        return data

    def inverse_scale_data(self, data):
        mean = self.statistics["mean"]
        std = self.statistics["stdev"]

        mean_out = mean[..., self.out_idxs]
        std_out = std[..., self.out_idxs]

        data.pred = data.pred * std_out + mean_out
        return data

    def inverse_route_features(self, data):
        raw = data.raw.clone()
        raw[:, 1, self.out_idxs] = data.pred
        data.reconstructed = raw
        return data

    def preprocess(self, data):
        data = self.route_features(data)
        data = self.scale_data(data)
        return data

    def postprocess(self, data):
        data = self.inverse_scale_data(data)
        data = self.inverse_route_features(data)
        return data

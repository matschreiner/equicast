import torch

from equicast import utils


class Scaler:
    def __init__(self, statistics):
        self.statistics = utils.cast_dict(statistics, torch.Tensor)
        self.std = self.statistics["stdev"]
        self.mean = self.statistics["mean"]

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

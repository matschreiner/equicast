import torch


class Scaler:
    """Z-score normalization scaler for weather data."""

    def __init__(self, statistics: dict[str, torch.Tensor]):
        self.statistics = statistics
        self.std = self.statistics["stdev"]
        self.mean = self.statistics["mean"]

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Scale data to normalized space (z-score normalization)."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Unscale data back to physical units."""
        return data * self.std + self.mean

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)

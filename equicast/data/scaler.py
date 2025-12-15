import torch


class Scaler(torch.nn.Module):
    """Z-score normalization scaler for weather data."""

    def __init__(self, statistics: dict[str, torch.Tensor]):
        super().__init__()
        self.statistics = statistics
        # Register as buffers so they move with .to(device)
        self.register_buffer("mean", self.statistics["mean"])
        self.register_buffer("std", self.statistics["stdev"])

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Scale data to normalized space (z-score normalization)."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Unscale data back to physical units."""
        return data * self.std + self.mean

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)

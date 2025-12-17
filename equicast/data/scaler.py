from abc import ABC, abstractmethod

import torch


class BaseScaler(torch.nn.Module, ABC):
    """Abstract base class for data scalers."""

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Scale data to normalized space."""
        pass

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Unscale data back to physical units."""
        pass

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)


class Scaler(BaseScaler):
    """Z-score normalization scaler for weather data."""

    def __init__(self, statistics: dict[str, torch.Tensor]):
        super().__init__()
        self.statistics = statistics
        # Register as buffers so they move with .to(device)
        self.register_buffer("mean", self.statistics["mean"])
        self.register_buffer("std", self.statistics["stdev"])

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Scale data to normalized space (z-score normalization)."""
        return (data - self.mean) / self.std  # type: ignore

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Unscale data back to physical units."""
        return data * self.std + self.mean  # type: ignore

    def transform_indices(
        self, data: torch.Tensor, indices: list[int]
    ) -> torch.Tensor:
        """Scale data using statistics for specific feature indices."""
        mean = self.mean[indices]
        std = self.std[indices]
        return (data - mean) / std

    def inverse_transform_indices(
        self, data: torch.Tensor, indices: list[int]
    ) -> torch.Tensor:
        """Unscale data using statistics for specific feature indices."""
        mean = self.mean[indices]
        std = self.std[indices]
        return data * std + mean

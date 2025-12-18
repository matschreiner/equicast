from abc import ABC, abstractmethod

import torch


class BaseNormalizer(torch.nn.Module, ABC):
    """Abstract base class for data normalizers."""

    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to normalized space."""
        pass

    @abstractmethod
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to physical units."""
        pass

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.transform(data)


class Normalizer(BaseNormalizer):
    """Z-score normalization normalizer for weather data."""

    def __init__(self, statistics: dict[str, torch.Tensor]):
        super().__init__()
        self.statistics = statistics
        # Register as buffers so they move with .to(device)
        self.register_buffer("mean", self.statistics["mean"])
        self.register_buffer("std", self.statistics["stdev"])

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to normalized space (z-score normalization)."""
        return (data - self.mean) / self.std  # type: ignore

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to physical units."""
        return data * self.std + self.mean  # type: ignore

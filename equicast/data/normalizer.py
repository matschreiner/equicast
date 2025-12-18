from abc import ABC, abstractmethod

import torch


class Normalizer(torch.nn.Module):
    """Z-score normalization normalizer for weather data."""

    def __init__(self, statistics: dict[str, torch.Tensor]):
        super().__init__()
        self.statistics = statistics
        # Register as buffers so they move with .to(device)
        self.register_buffer("mean", self.statistics["mean"])
        self.register_buffer("std", self.statistics["stdev"])

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize data to normalized space (z-score normalization)."""
        return (data - self.mean) / self.std  # type: ignore

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize data back to physical units."""
        return data * self.std + self.mean  # type: ignore

    def transform_indices(
        self, data: torch.Tensor, indices: list[int]
    ) -> torch.Tensor:
        """Normalize data using statistics at specified indices.

        Args:
            data: Tensor with shape [..., num_features] where num_features = len(indices)
            indices: List of feature indices to use for mean/std statistics

        Returns:
            Normalized tensor using mean[indices] and std[indices]
        """
        return (data - self.mean[indices]) / self.std[indices]  # type: ignore

    def inverse_transform_indices(
        self, data: torch.Tensor, indices: list[int]
    ) -> torch.Tensor:
        """Denormalize data using statistics at specified indices.

        Args:
            data: Tensor with shape [..., num_features] where num_features = len(indices)
            indices: List of feature indices to use for mean/std statistics

        Returns:
            Denormalized tensor using mean[indices] and std[indices]
        """
        return data * self.std[indices] + self.mean[indices]  # type: ignore

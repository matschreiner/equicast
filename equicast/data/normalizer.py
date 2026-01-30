from abc import ABC, abstractmethod

import numpy as np
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

    def compute_vector_statistics(
        self,
        dataset,  # anemoi dataset
        vector_indices: list[tuple[int, int]],  # [(u_idx, v_idx), ...]
        num_samples: int = 1000,
    ) -> None:
        """Compute mean vector norms from data samples.

        Args:
            dataset: Anemoi dataset to sample from
            vector_indices: List of (u_idx, v_idx) tuples for each vector
            num_samples: Number of random samples to use (max 1000)
        """

        num_samples = min(num_samples, len(dataset))
        sample_indices = np.random.choice(
            len(dataset), num_samples, replace=False
        )

        u_idxs, v_idxs = zip(*vector_indices)

        # Fetch all samples and compute norms
        data = dataset[sample_indices]
        u = data[..., u_idxs]
        v = data[..., v_idxs]
        norms = np.sqrt(u**2 + v**2)  # [num_samples, nodes, num_vectors]
        mean_norms = norms.mean(axis=(0, 1))  # [num_vectors]

        self.register_buffer("vector_mean_norm", torch.tensor(mean_norms))

    def transform_vector(
        self, data: torch.Tensor, vector_idx: int
    ) -> torch.Tensor:
        """Normalize a 2D vector so average norm is 1.

        Args:
            data: Tensor with shape [..., 2] containing (u, v) components
            vector_idx: Index into vector_mean_norm

        Returns:
            Normalized vector: data / mean_norm
        """
        return data / self.vector_mean_norm[vector_idx]

    def inverse_transform_vector(
        self, data: torch.Tensor, vector_idx: int
    ) -> torch.Tensor:
        """Denormalize a 2D vector back to physical units."""
        return data * self.vector_mean_norm[vector_idx]

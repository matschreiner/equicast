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


class VectorNormalizer(Normalizer):
    """Normalizer with additional vector normalization support."""

    def __init__(
        self,
        statistics: dict[str, torch.Tensor],
        dataset,  # anemoi dataset
        vector_indices: list[tuple[int, int]],  # [(u_idx, v_idx), ...]
        num_samples: int = 100,
    ):
        super().__init__(statistics)
        self._compute_vector_statistics(dataset, vector_indices, num_samples)

    def _compute_vector_statistics(
        self,
        dataset,  # anemoi dataset
        vector_indices: list[tuple[int, int]],  # [(u_idx, v_idx), ...]
        num_samples: int = 100,
    ) -> None:
        """Compute mean vector norms from data samples.

        Args:
            dataset: Anemoi dataset to sample from
            vector_indices: List of (u_idx, v_idx) tuples for each vector
            num_samples: Number of random samples to use
        """
        if not vector_indices:
            self.register_buffer("vector_mean_norm", torch.tensor([]))
            return

        num_samples = min(num_samples, len(dataset))
        sample_indices = np.random.choice(
            len(dataset), num_samples, replace=False
        )
        samples = dataset[sample_indices].squeeze()  # [samples, features, nodes]
        samples = samples.transpose(0, 2, 1)  # [samples, nodes, features]

        u_idxs, v_idxs = zip(*vector_indices)

        u = samples[..., u_idxs]
        v = samples[..., v_idxs]
        norms = np.sqrt(u**2 + v**2)  # [samples, nodes, num_vectors]

        # Mean over all dims except last (num_vectors)
        mean_norms = norms.mean(axis=tuple(range(norms.ndim - 1)))

        self.register_buffer("vector_mean_norm", torch.tensor(mean_norms))

    def transform_vectors(self, data: torch.Tensor) -> torch.Tensor:
        """Normalize vectors so average norm is 1.

        Args:
            data: Tensor with shape [..., num_vectors, 2]

        Returns:
            Normalized vectors: data / mean_norm
        """
        return data / self.vector_mean_norm.view(1, -1, 1)

    def inverse_transform_vectors(self, data: torch.Tensor) -> torch.Tensor:
        """Denormalize vectors back to physical units.

        Args:
            data: Tensor with shape [..., num_vectors, 2]

        Returns:
            Denormalized vectors: data * mean_norm
        """
        return data * self.vector_mean_norm.view(1, -1, 1)

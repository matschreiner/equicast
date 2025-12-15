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


class VectorAwareScaler(BaseScaler):
    """
    Scaler that handles both scalar and vector features.

    - Scalar features: standard z-score normalization
    - Vector features: magnitude-based normalization (rotation-equivariant)
    """

    def __init__(
        self,
        statistics: dict[str, torch.Tensor],
        name_to_index: dict[str, int],
        vector_features: dict[str, list[str]],
    ):
        super().__init__()
        self.statistics = statistics
        self.name_to_index = name_to_index
        self.vector_features = vector_features

        # Register scalar statistics as buffers
        self.register_buffer("mean", statistics["mean"].clone())
        self.register_buffer("std", statistics["stdev"].clone())

        # Compute and register magnitude statistics for vector features
        self._compute_vector_statistics()

    def _compute_vector_statistics(self):
        """
        Compute magnitude-based statistics for vector features.

        For rotation equivariance, we:
        1. Don't subtract mean (preserves direction)
        2. Normalize by magnitude std: σ_mag = √(σ_u² + σ_v² + ...)
        """
        # Create masks for vector vs scalar features
        n_features = len(self.mean)
        vector_mask = torch.zeros(n_features, dtype=torch.bool)
        magnitude_std = self.std.clone()

        for vector_name, component_names in self.vector_features.items():
            # Get indices for this vector's components
            component_indices = [
                self.name_to_index[name] for name in component_names
            ]

            # Mark these as vector components
            vector_mask[component_indices] = True

            # Compute magnitude std: √(σ₁² + σ₂² + ...)
            component_stds = self.std[component_indices]
            mag_std = torch.sqrt((component_stds**2).sum())

            # Use same magnitude std for all components of this vector
            magnitude_std[component_indices] = mag_std

        # Register buffers
        self.register_buffer("vector_mask", vector_mask)
        self.register_buffer("magnitude_std", magnitude_std)

        # For inverse transform, we need the original component stds
        self.register_buffer("component_std", self.std.clone())

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Scale data to normalized space.

        - Scalars: (x - μ) / σ
        - Vectors: x / σ_magnitude (rotation-equivariant)
        """
        scaled = (data - self.mean) / self.std

        if self.vector_mask.any():
            vector_scaled = (
                data[..., self.vector_mask]
                / self.magnitude_std[self.vector_mask]
            )
            scaled[..., self.vector_mask] = vector_scaled

        return scaled

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Unscale data back to physical units.

        - Scalars: x * σ + μ
        - Vectors: x * σ_magnitude
        """
        # Start with standard inverse transform
        unscaled = data * self.std + self.mean

        # Vector features: magnitude-based inverse (no mean addition)
        if self.vector_mask.any():
            # For vector components, use: x * σ_magnitude
            vector_unscaled = (
                data[..., self.vector_mask]
                * self.magnitude_std[self.vector_mask]
            )
            unscaled[..., self.vector_mask] = vector_unscaled

        return unscaled

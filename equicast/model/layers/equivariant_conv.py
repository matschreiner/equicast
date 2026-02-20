"""Equivariant message passing layers for 2D vector features."""

import torch
from torch import nn


class EquivariantLinear(nn.Module):
    """Linear layer for vector features (no bias to preserve equivariance)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply linear transformation to vectors.

        Args:
            x: [..., num_vectors, d]

        Returns:
            [..., out_features, d]
        """
        # Swap last two dims, apply linear, swap back
        return self.linear(x.transpose(-1, -2)).transpose(-1, -2)

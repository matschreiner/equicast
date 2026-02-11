"""Reusable embedding modules for equivariant models."""

import math

import torch
from torch import nn


class PositionalEmbedder(nn.Module):
    """Sinusoidal positional encoding for scalar distances.

    Maps a scalar distance to a vector of size `hidden_dim` using
    sinusoidal basis functions with geometrically spaced frequencies,
    normalised by `max_length`.
    """

    def __init__(self, hidden_dim: int, max_length: float = 10.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Frequencies: exp(-i * log(max_length) / (hidden_dim // 2))
        half = hidden_dim // 2
        freq = torch.exp(
            -torch.arange(half, dtype=torch.float) * math.log(max_length) / half
        )
        self.register_buffer("freq", freq)  # [half]

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Encode distances.

        Args:
            dist: [edges] or [edges, 1] scalar distances

        Returns:
            [edges, hidden_dim] positional encoding
        """
        dist = dist.view(-1, 1)  # [edges, 1]
        x = dist * self.freq  # [edges, half]
        return torch.cat([x.sin(), x.cos()], dim=-1)  # [edges, hidden_dim]

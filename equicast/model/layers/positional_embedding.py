"""Reusable embedding modules for equivariant models."""

import math

import torch
from torch import nn


class PositionalEmbedder(nn.Module):
    """Sinusoidal positional encoding for scalar distances.

    Maps a scalar distance to a vector of size `hidden_dim` using
    sinusoidal basis functions with log-spaced frequencies spanning
    from `1/max_length` to `1/min_length`.
    """

    def __init__(
        self,
        hidden_dim: int,
        max_length: float = math.pi,
        min_length: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        half = hidden_dim // 2
        freq = torch.exp(
            torch.linspace(
                math.log(1.0 / max_length),
                math.log(1.0 / min_length),
                half,
            )
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

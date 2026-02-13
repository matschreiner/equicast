"""Equivariant message passing layers for 2D vector features."""

from typing import Optional

import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing

from equicast.model.layers.mlp import MLP


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


class EquivariantMessagePassing(MessagePassing):
    """Equivariant message passing layer for scalar and vector features.

    Processes both invariant (scalar) and equivariant (vector) features
    while maintaining SO(2) equivariance for vectors.
    """

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        hidden_dim: Optional[int] = None,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)
        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim
        hidden_dim = hidden_dim or scalar_dim

        # Scalar message MLP: takes sender/receiver scalars + vector norms
        scalar_in_dim = 2 * scalar_dim + 2 * vector_dim
        self.scalar_mlp = MLP(
            in_dim=scalar_in_dim,
            out_dim=scalar_dim + vector_dim,  # scalar update + vector gates
            hidden_dim=hidden_dim,
        )

        # Vector message: linear combination of sender vectors
        self.vector_linear = EquivariantLinear(vector_dim, vector_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalar: [nodes, scalar_dim] invariant features
            vector: [nodes, vector_dim, 2] equivariant features
            edge_index: [2, edges] graph connectivity

        Returns:
            Updated (scalar, vector) features
        """
        from torch_geometric.utils import scatter

        src, dst = edge_index
        num_nodes = scalar.size(0)

        # Compute vector norms as invariant features
        vector_norm = vector.norm(dim=-1)  # [nodes, vector_dim]

        # Gather features for edges
        scalar_i, scalar_j = scalar[dst], scalar[src]
        vector_j = vector[src]
        vector_norm_i, vector_norm_j = vector_norm[dst], vector_norm[src]

        # Compute messages
        scalar_msg, vector_msg = self.message(scalar_i, scalar_j, vector_j, vector_norm_i, vector_norm_j)

        # Aggregate
        out_scalar = scatter(scalar_msg, dst, dim=0, dim_size=num_nodes, reduce=self.aggr)
        out_vector = scatter(vector_msg, dst, dim=0, dim_size=num_nodes, reduce=self.aggr)

        return out_scalar, out_vector

    def message(
        self,
        scalar_i: torch.Tensor,
        scalar_j: torch.Tensor,
        vector_j: torch.Tensor,
        vector_norm_i: torch.Tensor,
        vector_norm_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute messages from source to target nodes.

        Args:
            scalar_i: [edges, scalar_dim] target node scalars
            scalar_j: [edges, scalar_dim] source node scalars
            vector_j: [edges, vector_dim, 2] source node vectors
            vector_norm_i: [edges, vector_dim] target node vector norms
            vector_norm_j: [edges, vector_dim] source node vector norms

        Returns:
            (scalar_msg, vector_msg) messages
        """
        # Concatenate invariant features for scalar message
        scalar_in = torch.cat([scalar_i, scalar_j, vector_norm_i, vector_norm_j], dim=-1)

        # Compute scalar message and vector gates
        scalar_out = self.scalar_mlp(scalar_in)
        scalar_msg = scalar_out[..., : self.scalar_dim]
        vector_gates = scalar_out[..., self.scalar_dim :]  # [edges, vector_dim]

        # Compute gated vector message
        vector_transformed = self.vector_linear(vector_j)  # [edges, vector_dim, 2]
        vector_msg = vector_gates.unsqueeze(-1) * vector_transformed

        return scalar_msg, vector_msg


class EquivariantUpdate(nn.Module):
    """Update block that mixes scalar and vector information."""

    def __init__(self, scalar_dim: int, vector_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or scalar_dim

        # Vector transformations
        self.vector_U = EquivariantLinear(vector_dim, vector_dim)
        self.vector_V = EquivariantLinear(vector_dim, vector_dim)

        # Scalar MLP: takes scalars + vector norms
        self.scalar_mlp = MLP(
            in_dim=scalar_dim + vector_dim,
            out_dim=scalar_dim + 2 * vector_dim,  # scalar + vector_gate + vector_scale
            hidden_dim=hidden_dim,
        )

        self.scalar_dim = scalar_dim
        self.vector_dim = vector_dim

    def forward(self, scalar: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Update scalar and vector features.

        Args:
            scalar: [nodes, scalar_dim]
            vector: [nodes, vector_dim, 2]

        Returns:
            Updated (scalar, vector)
        """
        # Transform vectors
        Uv = self.vector_U(vector)  # [nodes, vector_dim, 2]
        Vv = self.vector_V(vector)  # [nodes, vector_dim, 2]

        # Compute invariants from Vv
        Vv_norm = Vv.norm(dim=-1)  # [nodes, vector_dim]

        # Scalar update path
        scalar_in = torch.cat([scalar, Vv_norm], dim=-1)
        scalar_out = self.scalar_mlp(scalar_in)

        d_scalar = scalar_out[..., : self.scalar_dim]
        vector_gate = scalar_out[..., self.scalar_dim : self.scalar_dim + self.vector_dim]
        vector_scale = scalar_out[..., self.scalar_dim + self.vector_dim :]

        # Vector update: gated Uv + contribution from Vv squared norm
        d_vector = vector_gate.unsqueeze(-1) * Uv

        # Scalar gets contribution from vector norms
        d_scalar = d_scalar + vector_scale * Vv_norm

        return scalar + d_scalar, vector + d_vector


class EquivariantBlock(nn.Module):
    """Combined message passing + update block."""

    def __init__(
        self,
        scalar_dim: int,
        vector_dim: int,
        hidden_dim: int = 64,
        aggr: str = "mean",
    ):
        super().__init__()
        self.message_passing = EquivariantMessagePassing(scalar_dim, vector_dim, hidden_dim, aggr)
        self.update = EquivariantUpdate(scalar_dim, vector_dim, hidden_dim)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalar: [nodes, scalar_dim]
            vector: [nodes, vector_dim, 2]
            edge_index: [2, edges]

        Returns:
            Updated (scalar, vector)
        """
        # Message passing with residual
        d_scalar, d_vector = self.message_passing(scalar, vector, edge_index)
        scalar = scalar + d_scalar
        vector = vector + d_vector

        # Update with residual (handled inside EquivariantUpdate)
        scalar, vector = self.update(scalar, vector)

        return scalar, vector

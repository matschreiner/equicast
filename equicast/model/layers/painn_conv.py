"""PaiNN-style equivariant message passing and update layers."""

import math

import torch
from torch import nn
from torch_geometric.utils import scatter

from equicast.model.layers.equivariant_conv import EquivariantLinear
from equicast.model.layers.mlp import MLP


class PositionalEncoder(nn.Module):
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


class PaiNN(nn.Module):
    """PaiNN backbone: embeds inputs, runs message passing + update, projects back."""

    def __init__(
        self,
        feature_config,
        grid_nodes: str = "grid",
        hidden_dim: int = 32,
        max_edge_length: float = 10.0,
        aggr: str = "mean",
    ):
        super().__init__()
        self.grid_nodes = grid_nodes

        scalar_in_dim = len(feature_config.forcing) + len(
            feature_config.prognostic
        )
        scalar_out_dim = len(feature_config.prognostic) + len(
            feature_config.diagnostic
        )
        vector_dim = len(feature_config.prognostic_vector)

        self.scalar_encoder = MLP(
            in_dim=scalar_in_dim, out_dim=hidden_dim, hidden_dim=hidden_dim
        )
        self.vector_encoder = EquivariantLinear(vector_dim, hidden_dim)
        self.positional_encoder = PositionalEncoder(
            hidden_dim, max_length=max_edge_length
        )

        self.block = PaiNNBlock(hidden_dim, aggr=aggr)

        self.scalar_decoder = MLP(
            in_dim=hidden_dim, out_dim=scalar_out_dim, hidden_dim=hidden_dim
        )
        self.vector_decoder = EquivariantLinear(hidden_dim, vector_dim)

    def forward(self, graph) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            graph: HeteroData with grid node features

        Returns:
            Dict with "scalar" and "vector" outputs (with residuals added)
        """
        scalar_in = graph[self.grid_nodes].input_scalar
        vector_in = graph[self.grid_nodes].input_vector
        edges = graph[self.grid_nodes, "to", self.grid_nodes]

        edge_index = edges["edge_index"]
        edge_dirs = edges.edge_dirs
        edge_length = edges.edge_length

        scalar_residual = graph[self.grid_nodes].residual_scalar
        vector_residual = graph[self.grid_nodes].residual_vector

        edge_pe = self.positional_encoder(edge_length)  # [edges, hidden_dim]

        # Encode
        scalar = self.scalar_encoder(scalar_in)
        vector = self.vector_encoder(vector_in)

        # Process
        scalar, vector = self.block(
            scalar, vector, edge_index, edge_dirs, edge_pe
        )

        # Decode
        scalar_out = self.scalar_decoder(scalar) + scalar_residual
        vector_out = self.vector_decoder(vector) + vector_residual

        return {"scalar": scalar_out, "vector": vector_out}


class PaiNNMessagePassing(nn.Module):
    """PaiNN-style equivariant message passing with edge direction/length filtering."""

    def __init__(
        self,
        hidden_dim: int,
        aggr: str = "mean",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.aggr = aggr

        # Node feature MLP: scalar_dim -> 3 * hidden_dim
        self.scalar_mlp = MLP(
            in_dim=hidden_dim, out_dim=3 * hidden_dim, hidden_dim=hidden_dim
        )
        # Edge filter MLP: pe_dim -> 3 * hidden_dim
        self.edge_mlp = MLP(
            in_dim=hidden_dim, out_dim=3 * hidden_dim, hidden_dim=hidden_dim
        )

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dirs: torch.Tensor,
        edge_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalar: [nodes, hidden_dim] invariant features
            vector: [nodes, hidden_dim, 2] equivariant features
            edge_index: [2, edges] graph connectivity
            edge_dirs: [edges, 2] unit direction vectors
            edge_pe: [edges, hidden_dim] positional encoding of edge lengths

        Returns:
            (d_scalar, d_vector) residual updates
        """
        src, dst = edge_index
        num_nodes = scalar.size(0)

        # Continuous filter: phi(scalar_j) * w(edge_pe)
        scalar_j = scalar[src]  # [edges, hidden_dim]
        edge_filter = self.edge_mlp(edge_pe)  # [edges, 3*hidden_dim]
        filtered = (
            self.scalar_mlp(scalar_j) * edge_filter
        )  # [edges, 3*hidden_dim]

        # Split into scalar message, edge-dir scale, vector-feature gate
        scalar_msg, scale_ed, scale_vf = filtered.chunk(3, dim=-1)

        # Vector messages: gated source vectors + scaled edge directions
        vector_j = vector[src]  # [edges, hidden_dim, 2]
        gated_vectors = (
            scale_vf.unsqueeze(-1) * vector_j
        )  # [edges, hidden_dim, 2]
        scaled_edge_dirs = scale_ed.unsqueeze(-1) * edge_dirs.unsqueeze(
            -2
        )  # [edges, hidden_dim, 2]
        vector_msg = gated_vectors + scaled_edge_dirs  # [edges, hidden_dim, 2]

        # Aggregate to destination nodes
        d_scalar = scatter(
            scalar_msg, dst, dim=0, dim_size=num_nodes, reduce=self.aggr
        )
        d_vector = scatter(
            vector_msg, dst, dim=0, dim_size=num_nodes, reduce=self.aggr
        )

        return d_scalar, d_vector


class PaiNNUpdate(nn.Module):
    """PaiNN-style equivariant update block."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.U = EquivariantLinear(hidden_dim, hidden_dim)
        self.V = EquivariantLinear(hidden_dim, hidden_dim)
        self.scalar_mlp = MLP(
            in_dim=2 * hidden_dim, out_dim=3 * hidden_dim, hidden_dim=hidden_dim
        )

    def forward(
        self, scalar: torch.Tensor, vector: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalar: [nodes, hidden_dim]
            vector: [nodes, hidden_dim, 2]

        Returns:
            Updated (scalar, vector)
        """
        Uv = self.U(vector)  # [nodes, hidden_dim, 2]
        Vv = self.V(vector)  # [nodes, hidden_dim, 2]

        Vv_norm = Vv.norm(dim=-1)  # [nodes, hidden_dim]

        scalar_in = torch.cat([scalar, Vv_norm], dim=-1)  # [nodes, 2*hidden_dim]
        s1, s2, s3 = self.scalar_mlp(scalar_in).chunk(3, dim=-1)

        # Vector update: gate Uv
        d_vector = s1.unsqueeze(-1) * Uv  # [nodes, hidden_dim, 2]

        # Scalar update: squared norm contribution + bias
        Vv_squared_norm = Vv_norm**2  # [nodes, hidden_dim]
        d_scalar = s2 * Vv_squared_norm + s3

        return scalar + d_scalar, vector + d_vector


class PaiNNBlock(nn.Module):
    """Combined PaiNN message passing + update block."""

    def __init__(
        self,
        hidden_dim: int,
        vector_dim: int | None = None,
        aggr: str = "mean",
    ):
        super().__init__()
        vector_dim = vector_dim or hidden_dim

        # Project vectors from vector_dim -> hidden_dim if needed
        self.vector_proj_in = (
            EquivariantLinear(vector_dim, hidden_dim)
            if vector_dim != hidden_dim
            else nn.Identity()
        )
        self.vector_proj_out = (
            EquivariantLinear(hidden_dim, vector_dim)
            if vector_dim != hidden_dim
            else nn.Identity()
        )

        self.message_passing = PaiNNMessagePassing(hidden_dim, aggr)
        self.update = PaiNNUpdate(hidden_dim)
        self.vector_dim = vector_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        edge_index: torch.Tensor,
        edge_dirs: torch.Tensor,
        edge_pe: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            scalar: [nodes, hidden_dim]
            vector: [nodes, vector_dim, 2]
            edge_index: [2, edges]
            edge_dirs: [edges, 2]
            edge_pe: [edges, hidden_dim] positional encoding of edge lengths

        Returns:
            Updated (scalar, vector) with original dimensions
        """
        # Project vectors: nv -> nh
        vector_h = self.vector_proj_in(vector)  # [nodes, hidden_dim, 2]

        # Message passing with residual
        d_scalar, d_vector = self.message_passing(
            scalar, vector_h, edge_index, edge_dirs, edge_pe
        )
        scalar = scalar + d_scalar
        vector_h = vector_h + d_vector

        # Update (residual handled inside PaiNNUpdate)
        scalar, vector_h = self.update(scalar, vector_h)

        # Project vectors back: nh -> nv, with residual
        vector = vector + self.vector_proj_out(vector_h)

        return scalar, vector

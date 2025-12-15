"""GraphCast-style Encoder-Processor-Decoder architecture."""

from typing import Optional

import torch
from torch_geometric.nn.conv import MessagePassing

from equicast.model.layers.mlp import MLP


class EncProcDec(torch.nn.Module):
    """
    Encoder-Processor-Decoder architecture for graph-based weather forecasting.

    Architecture:
        1. Encoder: Maps grid features to mesh latent space
        2. Processor: Stack of GNN layers on mesh
        3. Decoder: Maps mesh features back to grid predictions

    This follows the GraphCast design pattern.
    """

    def __init__(
        self,
        feature_config,
        hidden_dim: int = 256,
        num_processor_layers: int = 16,
        grid_nodes: str = "grid",
        mesh_nodes: str = "mesh",
    ):
        super().__init__()

        # Calculate input/output dimensions from feature config
        in_dim = len(feature_config.forcing) + len(feature_config.prognostic)
        out_dim = len(feature_config.prognostic) + len(feature_config.diagnostic)

        self.grid_nodes = grid_nodes
        self.mesh_nodes = mesh_nodes

        # Encoder: Grid -> Mesh
        self.encoder = GridToMeshEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
        )

        # Processor: Mesh -> Mesh (stack of GNN layers)
        self.processor_layers = torch.nn.ModuleList(
            [
                MeshProcessor(hidden_dim=hidden_dim)
                for _ in range(num_processor_layers)
            ]
        )

        # Decoder: Mesh -> Grid
        self.decoder = MeshToGridDecoder(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
        )

    def forward(self, graph):
        """
        Forward pass through encoder-processor-decoder.

        Args:
            graph: Heterogeneous graph with:
                - graph["grid"].cond: Grid input features [N_grid, in_dim]
                - graph["grid", "to", "mesh"]: Grid-to-mesh edges
                - graph["mesh", "to", "mesh"]: Mesh edges
                - graph["mesh", "to", "grid"]: Mesh-to-grid edges

        Returns:
            Grid predictions [N_grid, out_dim]
        """
        # Encode: Grid -> Mesh
        mesh_features = self.encoder(
            grid_features=graph[self.grid_nodes].cond,
            edge_storage=graph[self.grid_nodes, "to", self.mesh_nodes],
        )

        # Process: Mesh -> Mesh (multiple layers with residuals)
        for processor_layer in self.processor_layers:
            mesh_features = mesh_features + processor_layer(
                x=mesh_features,
                edge_storage=graph[self.mesh_nodes, "to", self.mesh_nodes],
            )

        # Decode: Mesh -> Grid
        grid_pred = self.decoder(
            mesh_features=mesh_features,
            edge_storage=graph[self.mesh_nodes, "to", self.grid_nodes],
        )

        return grid_pred


class GridToMeshEncoder(MessagePassing):
    """Encode grid features to mesh latent space."""

    def __init__(self, in_dim: int, hidden_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.mlp = MLP(
            in_dim=in_dim,
            out_dim=hidden_dim,
            num_layers=3,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        grid_features: torch.Tensor,
        edge_storage: dict,
    ) -> torch.Tensor:
        """
        Aggregate grid features to mesh nodes.

        Args:
            grid_features: Grid node features [N_grid, in_dim]
            edge_storage: Grid-to-mesh edges

        Returns:
            Mesh node features [N_mesh, hidden_dim]
        """
        edge_index = edge_storage["edge_index"].long()

        # Propagate from grid (source) to mesh (target)
        mesh_features = self.propagate(
            x=grid_features,
            edge_index=edge_index,
        )

        return mesh_features

    def message(self, x_j):  # type: ignore
        """Process features from source grid nodes."""
        return self.mlp(x_j)


class MeshProcessor(MessagePassing):
    """Process features on the mesh with message passing."""

    def __init__(self, hidden_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.mlp = MLP(
            in_dim=2 * hidden_dim,
            out_dim=hidden_dim,
            num_layers=3,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_storage: dict,
    ) -> torch.Tensor:
        """
        Message passing on mesh.

        Args:
            x: Mesh node features [N_mesh, hidden_dim]
            edge_storage: Mesh-to-mesh edges

        Returns:
            Updated mesh features [N_mesh, hidden_dim]
        """
        edge_index = edge_storage["edge_index"].long()

        out = self.propagate(
            x=x,
            edge_index=edge_index,
        )

        return out

    def message(self, x_j, x_i):  # type: ignore
        """Compute messages using source and target node features."""
        return self.mlp(torch.cat([x_i, x_j], dim=-1))


class MeshToGridDecoder(MessagePassing):
    """Decode mesh features back to grid predictions."""

    def __init__(self, hidden_dim: int, out_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.mlp = MLP(
            in_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=3,
            hidden_dim=hidden_dim,
        )

    def forward(
        self,
        mesh_features: torch.Tensor,
        edge_storage: dict,
    ) -> torch.Tensor:
        """
        Aggregate mesh features to grid nodes.

        Args:
            mesh_features: Mesh node features [N_mesh, hidden_dim]
            edge_storage: Mesh-to-grid edges

        Returns:
            Grid predictions [N_grid, out_dim]
        """
        edge_index = edge_storage["edge_index"].long()

        # Propagate from mesh (source) to grid (target)
        grid_pred = self.propagate(
            x=mesh_features,
            edge_index=edge_index,
        )

        return grid_pred

    def message(self, x_j):  # type: ignore
        """Process features from source mesh nodes."""
        return self.mlp(x_j)

"""GraphCast-style Encoder-Processor-Decoder architecture."""

from typing import Optional

import torch
from torch_geometric.nn.conv import MessagePassing

from equicast.model.layers.mlp import MLP


class Graphcast(torch.nn.Module):
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
        data_handler,
        hidden_dim: int = 256,
        num_processor_layers: int = 16,
        num_input_steps: int = 2,
        grid_nodes: str = "grid",
        mesh_nodes: str = "mesh",
        grid2mesh_edge_dim: int = 0,
        mesh2mesh_edge_dim: int = 0,
        mesh2grid_edge_dim: int = 0,
    ):
        super().__init__()
        self.data_handler = data_handler

        # Calculate input/output dimensions
        # Input: num_input_steps timesteps concatenated (GraphCast uses 2 steps)
        in_dim = num_input_steps * data_handler.in_dim
        out_dim = data_handler.out_dim

        self.grid_nodes = grid_nodes
        self.mesh_nodes = mesh_nodes

        # Encoder: Grid -> Mesh
        self.encoder = GridToMeshEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            edge_dim=grid2mesh_edge_dim,
        )

        # Processor: Mesh -> Mesh (stack of GNN layers)
        self.processor_layers = torch.nn.ModuleList(
            [MeshProcessor(hidden_dim=hidden_dim, edge_dim=mesh2mesh_edge_dim) for _ in range(num_processor_layers)]
        )

        # Decoder: Mesh -> Grid
        self.decoder = MeshToGridDecoder(
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            edge_dim=mesh2grid_edge_dim,
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
            grid_features=graph[self.grid_nodes].input,
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

        return grid_pred + graph[self.grid_nodes].residual


class GridToMeshEncoder(MessagePassing):
    """
    Encode grid features to mesh latent space with learned edge weighting.

    Uses distance-weighted aggregation: Σ w_ij * φ(x_j, e_ij)
    where w_ij are learned from edge features and softmax-normalized.
    """

    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int = 0, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.edge_dim = edge_dim

        # Message MLP: φ(x_j, e_ij)
        mlp_in_dim = in_dim + edge_dim
        self.message_mlp = MLP(
            in_dim=mlp_in_dim,
            out_dim=hidden_dim,
            num_layers=3,
            hidden_dim=hidden_dim,
        )

        # Edge weight MLP: learns w_ij from edge features (dx, dy)
        if edge_dim > 0:
            self.edge_weight_mlp = MLP(
                in_dim=edge_dim,
                out_dim=1,
                num_layers=2,
                hidden_dim=hidden_dim // 2,
            )
        else:
            self.edge_weight_mlp = None

    def forward(
        self,
        grid_features: torch.Tensor,
        edge_storage: dict,
    ) -> torch.Tensor:
        """
        Aggregate grid features to mesh nodes with learned edge weights.

        Args:
            grid_features: Grid node features [N_grid, in_dim]
            edge_storage: Grid-to-mesh edges with edge_attr (dx, dy)

        Returns:
            Mesh node features [N_mesh, hidden_dim]
        """
        edge_index = edge_storage["edge_index"].long()
        edge_attr = edge_storage.get("edge_attr", None)

        # Propagate from grid (source) to mesh (target)
        mesh_features = self.propagate(
            x=grid_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return mesh_features

    def message(self, x_j, edge_attr, index, ptr, size_i):  # type: ignore
        """
        Compute weighted messages: w_ij * φ(x_j, e_ij).

        Edge weights are learned from spatial features (dx, dy) and
        softmax-normalized per target node.
        """
        if edge_attr is not None and self.edge_dim > 0:
            # Compute message features φ(x_j, e_ij)
            msg_input = torch.cat([x_j, edge_attr], dim=-1)
            msg = self.message_mlp(msg_input)

            # Compute edge weights from spatial features
            edge_weights = self.edge_weight_mlp(edge_attr)  # [num_edges, 1]

            # Softmax normalization per target node
            from torch_geometric.utils import softmax

            edge_weights = softmax(edge_weights, index, ptr, size_i)

            # Apply weights: w_ij * φ(x_j, e_ij)
            return edge_weights * msg
        else:
            msg_input = x_j
            return self.message_mlp(msg_input)


class MeshProcessor(MessagePassing):
    """
    Process features on the mesh with weighted message passing.

    Uses distance-weighted aggregation: Σ w_ij * φ(x_i, x_j, e_ij)
    where w_ij are learned from edge features and softmax-normalized.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 0, aggr: str = "add"):
        super().__init__(aggr=aggr)
        self.edge_dim = edge_dim

        # Message MLP: φ(x_i, x_j, e_ij)
        mlp_in_dim = 2 * hidden_dim + edge_dim
        self.message_mlp = MLP(
            in_dim=mlp_in_dim,
            out_dim=hidden_dim,
            num_layers=3,
            hidden_dim=hidden_dim,
        )

        # Edge weight MLP: learns w_ij from edge features (dx, dy)
        if edge_dim > 0:
            self.edge_weight_mlp = MLP(
                in_dim=edge_dim,
                out_dim=1,
                num_layers=2,
                hidden_dim=hidden_dim // 2,
            )
        else:
            self.edge_weight_mlp = None

    def forward(
        self,
        x: torch.Tensor,
        edge_storage: dict,
    ) -> torch.Tensor:
        """
        Message passing on mesh with learned edge weights.

        Args:
            x: Mesh node features [N_mesh, hidden_dim]
            edge_storage: Mesh-to-mesh edges with edge_attr (dx, dy)

        Returns:
            Updated mesh features [N_mesh, hidden_dim]
        """
        edge_index = edge_storage["edge_index"].long()
        edge_attr = edge_storage.get("edge_attr", None)

        out = self.propagate(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return out

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):  # type: ignore
        """
        Compute weighted messages: w_ij * φ(x_i, x_j, e_ij).

        Edge weights are learned from spatial features (dx, dy) and
        softmax-normalized per target node.
        """
        # Compute message features φ(x_i, x_j, e_ij)
        if edge_attr is not None and self.edge_dim > 0:
            msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
            msg = self.message_mlp(msg_input)

            # Compute edge weights from spatial features
            edge_weights = self.edge_weight_mlp(edge_attr)  # [num_edges, 1]

            # Softmax normalization per target node
            from torch_geometric.utils import softmax

            edge_weights = softmax(edge_weights, index, ptr, size_i)

            # Apply weights: w_ij * φ(x_i, x_j, e_ij)
            return edge_weights * msg
        else:
            msg_input = torch.cat([x_i, x_j], dim=-1)
            return self.message_mlp(msg_input)


class MeshToGridDecoder(MessagePassing):
    """
    Decode mesh features back to grid predictions with learned edge weighting.

    Uses distance-weighted aggregation: Σ w_ij * φ(x_j, e_ij)
    where w_ij are learned from edge features and softmax-normalized.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        edge_dim: int = 0,
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)
        self.edge_dim = edge_dim

        # Message MLP: φ(x_j, e_ij)
        mlp_in_dim = hidden_dim + edge_dim
        self.message_mlp = MLP(
            in_dim=mlp_in_dim,
            out_dim=out_dim,
            num_layers=3,
            hidden_dim=hidden_dim,
        )

        # Edge weight MLP: learns w_ij from edge features (dx, dy)
        if edge_dim > 0:
            self.edge_weight_mlp = MLP(
                in_dim=edge_dim,
                out_dim=1,
                num_layers=2,
                hidden_dim=hidden_dim // 2,
            )
        else:
            self.edge_weight_mlp = None

    def forward(
        self,
        mesh_features: torch.Tensor,
        edge_storage: dict,
    ) -> torch.Tensor:
        """
        Aggregate mesh features to grid nodes with learned edge weights.

        Args:
            mesh_features: Mesh node features [N_mesh, hidden_dim]
            edge_storage: Mesh-to-grid edges with edge_attr (dx, dy)

        Returns:
            Grid predictions [N_grid, out_dim]
        """
        edge_index = edge_storage["edge_index"].long()
        edge_attr = edge_storage.get("edge_attr", None)

        # Propagate from mesh (source) to grid (target)
        grid_pred = self.propagate(
            x=mesh_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

        return grid_pred

    def message(self, x_j, edge_attr, index, ptr, size_i):  # type: ignore
        """
        Compute weighted messages: w_ij * φ(x_j, e_ij).

        Edge weights are learned from spatial features (dx, dy) and
        softmax-normalized per target node.
        """
        if edge_attr is not None and self.edge_dim > 0:
            # Compute message features φ(x_j, e_ij)
            msg_input = torch.cat([x_j, edge_attr], dim=-1)
            msg = self.message_mlp(msg_input)

            # Compute edge weights from spatial features
            edge_weights = self.edge_weight_mlp(edge_attr)  # [num_edges, 1]

            # Softmax normalization per target node
            from torch_geometric.utils import softmax

            edge_weights = softmax(edge_weights, index, ptr, size_i)

            # Apply weights: w_ij * φ(x_j, e_ij)
            return edge_weights * msg
        else:
            msg_input = x_j
            return self.message_mlp(msg_input)

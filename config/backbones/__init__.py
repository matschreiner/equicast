from equicast.model.backbones.gnn import GNN
from equicast.model.backbones.graphcast import GraphCast
from equicast.model.backbones.painn import PaiNN


def build_painn(data_handler, edges, input_nodes="grid", hidden_dim=128, aggr="mean"):
    return PaiNN(
        in_dim=data_handler.in_dim,
        out_dim=data_handler.out_dim,
        in_vector_dim=data_handler.in_vector_dim,
        out_vector_dim=data_handler.out_vector_dim,
        edges=edges,
        input_nodes=input_nodes,
        hidden_dim=hidden_dim,
        aggr=aggr,
    )


def build_gnn(data_handler, edges, input_nodes="grid", hidden_dim=64, aggr="mean", edge_dir_dim=2):
    return GNN(
        in_dim=data_handler.in_dim,
        out_dim=data_handler.out_dim,
        edges=edges,
        input_nodes=input_nodes,
        hidden_dim=hidden_dim,
        aggr=aggr,
        edge_dir_dim=edge_dir_dim,
    )


def build_graphcast(data_handler, edges, input_nodes="grid", hidden_dim=64, aggr="sum"):
    return GraphCast(
        in_dim=data_handler.in_dim,
        out_dim=data_handler.out_dim,
        edges=edges,
        input_nodes=input_nodes,
        hidden_dim=hidden_dim,
        aggr=aggr,
    )

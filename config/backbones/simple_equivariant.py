import fiddle as fdl
import torch

from equicast.data import EquivariantGraphDataHandler, FeatureConfig
from equicast.model.backbones.simple_equivariant_gnn import SimpleEquivariantGNN
from equicast.model.model import equivariant_loss_fn

torch.set_float32_matmul_precision("medium")


def backbone_config(dataset_path):
    feature_config = fdl.Config(
        FeatureConfig.from_yaml, path="hydraconfig/features/base_equivariant.yaml"
    )
    data_handler = fdl.Config(
        EquivariantGraphDataHandler, feature_config=feature_config, dataset_path=dataset_path
    )

    backbone = fdl.Config(
        SimpleEquivariantGNN,
        data_handler=data_handler,
        grid_nodes="grid",
    )

    return backbone, equivariant_loss_fn

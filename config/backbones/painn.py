import fiddle as fdl
import torch

from equicast.data import EquivariantGraphDataHandler, FeatureConfig
from equicast.model.backbones.painn import PaiNN
from equicast.model.model import Model, equivariant_loss_fn

torch.set_float32_matmul_precision("medium")


def backbone_config(dataset_path):
    feature_config = fdl.Config(
        FeatureConfig.from_yaml, path="hydraconfig/features/base_equivariant.yaml"
    )
    data_handler = fdl.Config(
        EquivariantGraphDataHandler, feature_config=feature_config, dataset_path=dataset_path
    )

    backbone = fdl.Config(
        PaiNN,
        data_handler=data_handler,
        edges=[
            ("grid", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "grid"),
        ],
        input_nodes="grid",
        hidden_dim=128,
    )

    return fdl.Config(
        Model,
        backbone=backbone,
        loss_fn=equivariant_loss_fn,
        compile_backbone=True,
    )

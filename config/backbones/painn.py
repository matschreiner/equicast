import fiddle as fdl
import torch

from equicast.model.backbones.painn import PaiNN

torch.set_float32_matmul_precision("medium")


def backbone_config(feature_indices):
    return fdl.Config(
        PaiNN,
        feature_indices=feature_indices,
        edges=[
            ("grid", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "grid"),
        ],
        input_nodes="grid",
        hidden_dim=128,
    )

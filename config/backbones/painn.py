import fiddle as fdl
import torch

from equicast.model.backbones.painn import PaiNN

torch.set_float32_matmul_precision("medium")


def backbone_config(data_handler):
    return fdl.Config(
        PaiNN,
        data_handler=data_handler,
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

import fiddle as fdl

from equicast.logger import MLFlowLogger
from equicast.model.backbones.painn import PaiNN


def fiddler(cfg: fdl.Config) -> None:
    cfg.model.backbone = fdl.Config(
        PaiNN,
        data_handler=cfg.model.backbone.data_handler,
        edges=[
            ("grid", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "grid"),
        ],
        input_nodes="grid",
        hidden_dim=256,
    )

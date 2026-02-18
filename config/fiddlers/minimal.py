import fiddle as fdl

from equicast.logger import MLFlowLogger
from equicast.model.backbones.painn import PaiNN
from equicast.model.schedulers import WarmupCosineAnnealingLR


def fiddler(cfg: fdl.Config) -> None:
    cfg.model.scheduler_factory = fdl.Partial(
        WarmupCosineAnnealingLR,
        warmup_steps=10,
        total_steps=100,
    )

    cfg.model.backbone = fdl.Config(
        PaiNN,
        data_handler=cfg.model.backbone.data_handler,
        edges=[
            ("grid", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "grid"),
        ],
        input_nodes="grid",
        hidden_dim=4,
    )

    cfg.dataloader.num_workers = 0

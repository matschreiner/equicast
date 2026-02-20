import fiddle as fdl

from equicast.model.schedulers import WarmupCosineAnnealingLR


def fiddler(cfg: fdl.Config) -> None:
    cfg.model.backbone.hidden_dim = int(256 * 1.5)

    cfg.model.backbone.edges = [
        ("grid", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "grid"),
    ]

    cfg.model.scheduler_factory = fdl.Partial(
        WarmupCosineAnnealingLR,
        warmup_steps=10000,
        total_steps=450000,
    )

import fiddle as fdl

from equicast.model.schedulers import WarmupCosineAnnealingLR


def fiddler(cfg: fdl.Config) -> None:
    cfg.dataloader.batch_size = 4

    cfg.model.scheduler_factory = fdl.Partial(
        WarmupCosineAnnealingLR,
        warmup_steps=5000,
        total_steps=220000,
    )

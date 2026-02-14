from functools import partial

import fiddle as fdl

from equicast.model.schedulers import WarmupCosineAnnealingLR


def fiddler(cfg: fdl.Config) -> None:
    cfg.model.scheduler_factory = partial(
        WarmupCosineAnnealingLR,
        warmup_steps=10000,
        total_steps=864000,
    )

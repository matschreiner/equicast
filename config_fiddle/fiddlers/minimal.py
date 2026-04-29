import fiddle as fdl

from equicast.logger import MLFlowLogger
from equicast.model.schedulers import WarmupCosineAnnealingLR


def fiddler(cfg: fdl.Config) -> None:
    cfg.model.scheduler_factory = fdl.Partial(
        WarmupCosineAnnealingLR,
        warmup_steps=10,
        total_steps=100,
    )

    cfg.model.backbone.hidden_dim = 4
    cfg.dataloader.num_workers = 0
    cfg.logger.experiment_name = "minimal"

import fiddle as fdl

from equicast.logger import MLFlowLogger


def fiddler(cfg: fdl.Config) -> None:
    logger = fdl.Config(
        MLFlowLogger,
        tracking_uri="file:./mlruns",
        experiment_name="trash",
    )
    cfg.logger = logger
    cfg.trainer.logger = logger

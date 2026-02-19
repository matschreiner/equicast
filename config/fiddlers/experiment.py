import fiddle as fdl

from equicast import TRACKING_URI
from equicast.logger import MLFlowLogger


def fiddler(cfg: fdl.Config, name: str) -> None:
    logger = fdl.Config(
        MLFlowLogger,
        tracking_uri=TRACKING_URI,
        experiment_name=name,
    )
    cfg.logger = logger
    cfg.trainer.logger = logger

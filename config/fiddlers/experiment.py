import fiddle as fdl

from equicast.logger import MLFlowLogger


def fiddler(cfg: fdl.Config, name: str) -> None:
    logger = fdl.Config(
        MLFlowLogger,
        tracking_uri="sqlite:///mlflow/mlflow.db",
        experiment_name=name,
    )
    cfg.logger = logger
    cfg.trainer.logger = logger

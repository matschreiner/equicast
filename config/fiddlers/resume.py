import fiddle as fdl

from equicast.logger import MLFlowLogger


def fiddler(cfg: fdl.Config, run_id: str) -> None:
    cfg.logger = fdl.Config(
        MLFlowLogger,
        tracking_uri="file:./mlruns",
        experiment_name="continue",
    )

    cfg.logger.run_id = run_id

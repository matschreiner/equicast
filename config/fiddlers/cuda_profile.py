import fiddle as fdl

from equicast.profiler import CUDAProfiler


def fiddler(cfg: fdl.Config) -> None:
    cfg.trainer.profiler = fdl.Config(
        CUDAProfiler,
        logger=cfg.logger,
        wait=5,
        warmup=5,
        active=20,
    )

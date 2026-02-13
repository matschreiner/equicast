import fiddle as fdl


def fiddler(cfg: fdl.Config) -> None:
    cfg.trainer.max_steps = 10
    cfg.trainer.overfit_batches = 1
    cfg.model.compile_backbone = False

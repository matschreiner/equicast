import fiddle as fdl


def fiddler(cfg: fdl.Config) -> None:
    cfg.dataloader.batch_size = 4

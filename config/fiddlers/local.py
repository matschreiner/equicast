import fiddle as fdl


def fiddler(cfg: fdl.Config) -> None:
    cfg.dataloader.num_workers = 0
    cfg.dataloader.pin_memory = False

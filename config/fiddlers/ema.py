import fiddle as fdl

from equicast.callbacks import EMA


def fiddler(cfg: fdl.Config, decay: float = 0.999) -> None:
    ema_callback = fdl.Config(EMA, decay=decay)
    cfg.trainer.callbacks.append(ema_callback)

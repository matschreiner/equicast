import fiddle as fdl

from equicast.callbacks import EMA


def fiddler(cfg: fdl.Config, decay: float = 0.999, update_starting_at_step: int = 1000) -> None:
    ema_callback = fdl.Config(EMA, decay=decay, update_starting_at_step=update_starting_at_step)
    cfg.trainer.callbacks.append(ema_callback)

import fiddle as fdl

from equicast.callbacks import EMA


def fiddler(cfg: fdl.Config, decay: float = 0.999, update_starting_at_step: int = 100000) -> None:
    every_n = 100
    start = 0  # 100000
    ema_callback = fdl.Config(EMA, decay=decay, update_starting_at_step=start, update_every_n_steps=every_n)
    cfg.trainer.callbacks.append(ema_callback)

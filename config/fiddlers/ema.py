from functools import partial

import fiddle as fdl
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


def fiddler(cfg: fdl.Config, decay: float = 0.999) -> None:
    cfg.model.ema_decay = decay

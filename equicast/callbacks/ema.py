from torch.optim.swa_utils import get_ema_multi_avg_fn

from lightning.pytorch.callbacks import WeightAveraging


class EMA(WeightAveraging):
    """Exponential Moving Average callback using Lightning's WeightAveraging."""

    def __init__(self, decay: float = 0.999, update_after: int = 10000, **kwargs):
        super().__init__(multi_avg_fn=get_ema_multi_avg_fn(decay), **kwargs)
        self.update_after = update_after

    def should_update(self, step_idx=None, epoch_idx=None):
        return step_idx is not None and step_idx >= self.update_after

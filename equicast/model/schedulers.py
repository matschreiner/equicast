"""Custom learning rate schedulers."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class WarmupCosineAnnealingLR(SequentialLR):
    """
    Learning rate scheduler with linear warmup followed by cosine annealing.

    This is commonly used in large model training (e.g., GraphCast, Transformers).

    The learning rate schedule works as follows:
    1. Linear warmup: LR increases linearly from start_factor * base_lr to base_lr
       over warmup_steps
    2. Cosine annealing: LR decreases following a cosine curve from base_lr to
       eta_min over (total_steps - warmup_steps)

    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of steps for linear warmup phase
        total_steps: Total number of training steps
        eta_min: Minimum learning rate (default: 1e-7)
        start_factor: Starting LR multiplier for warmup (default: 0.01)
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> optimizer = Adam(model.parameters(), lr=1e-4)
        >>> scheduler = WarmupCosineAnnealingLR(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     total_steps=10000,
        ...     eta_min=1e-7,
        ... )
        >>> for epoch in range(num_epochs):
        ...     train(...)
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 1e-7,
        start_factor: float = 0.01,
        last_epoch: int = -1,
    ):
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})"
            )

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.start_factor = start_factor

        # Phase 1: Linear warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
            last_epoch=last_epoch,
        )

        # Phase 2: Cosine annealing
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )

        # Chain them together
        super().__init__(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
            last_epoch=last_epoch,
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"warmup_steps={self.warmup_steps}, "
            f"total_steps={self.total_steps}, "
            f"eta_min={self.eta_min}, "
            f"start_factor={self.start_factor})"
        )

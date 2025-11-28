from typing import Any

from hydra.utils import get_class


def from_checkpoint(model_class: str, checkpoint_provider, *args, **kwargs: Any):
    cls = get_class(model_class)
    ckpt = checkpoint_provider.get_checkpoint()
    return cls.load_from_checkpoint(ckpt)

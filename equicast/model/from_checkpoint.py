from typing import Any

from hydra.utils import get_class


def from_checkpoint(model_cls: str, checkpoint_provider, *args, **kwargs: Any):
    ckpt = checkpoint_provider.get_checkpoint()
    return model_cls.load_from_checkpoint(ckpt)

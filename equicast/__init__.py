import lovely_tensors as lt
import torch

from equicast.utils.namegen import cute

lt.monkey_patch()

DTYPE = torch.float32
CHECKPOINT_PATH = "checkpoint"

__all__ = ["cute", "DTYPE", "CHECKPOINT_PATH"]

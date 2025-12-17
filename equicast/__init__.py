import lovely_tensors as lt
import torch

lt.monkey_patch()

DTYPE = torch.float32
CHECKPOINT_PATH = "checkpoint"

__all__ = ["DTYPE", "CHECKPOINT_PATH"]

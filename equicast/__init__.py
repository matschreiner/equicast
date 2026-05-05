import torch

torch.set_float32_matmul_precision("medium")

DTYPE = torch.float32

__all__ = ["DTYPE"]

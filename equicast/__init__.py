import mlflow
import lovely_tensors as lt
import torch

lt.monkey_patch()
torch.set_float32_matmul_precision("medium")

DTYPE = torch.float32
CHECKPOINT_PATH = "checkpoints"
TRACKING_URI = "mlflow"

mlflow.set_tracking_uri(TRACKING_URI)

__all__ = ["DTYPE", "CHECKPOINT_PATH", "TRACKING_URI"]

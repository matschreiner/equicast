import mlflow
import lovely_tensors as lt
import torch

lt.monkey_patch()

DTYPE = torch.float32
CHECKPOINT_PATH = "checkpoints"
TRACKING_URI = "sqlite:///mlflow/mlflow.db"

mlflow.set_tracking_uri(TRACKING_URI)

__all__ = ["DTYPE", "CHECKPOINT_PATH", "TRACKING_URI"]

import yaml
import mlflow
import lovely_tensors as lt
import torch
from mlflow.store.tracking.file_store import FileStore as _FileStore

lt.monkey_patch()
torch.set_float32_matmul_precision("medium")

DTYPE = torch.float32

with open("config/config.yaml") as _f:
    _cfg = yaml.safe_load(_f)
    TRACKING_URI = _cfg["tracking_uri"]
    CHECKPOINT_PATH = _cfg["checkpoint_dir"]

mlflow.set_tracking_uri(TRACKING_URI)


def _create_experiment_named(self, name, artifact_location=None, tags=None):
    """Use experiment name as ID for portable, human-readable experiment directories."""
    self._check_root_dir()
    self._validate_experiment_does_not_exist(name)
    return self._create_experiment_with_id(name, name, artifact_location, tags)


_FileStore.create_experiment = _create_experiment_named

__all__ = ["DTYPE", "CHECKPOINT_PATH", "TRACKING_URI"]

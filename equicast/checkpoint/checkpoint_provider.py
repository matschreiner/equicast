import os
from abc import ABC, abstractmethod

from equicast import CHECKPOINT_PATH


class CheckpointProvider(ABC):
    @abstractmethod
    def get_checkpoint(self) -> str:
        """Loads the checkpoint data from the specified path."""
        pass


class MLFlowCheckpointProvider(CheckpointProvider):
    def __init__(self, tracking_uri: str, run_id: str, checkpoint_name: str):
        from mlflow.tracking import MlflowClient

        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.run_id = run_id

        self.checkpoint_name = (
            checkpoint_name if checkpoint_name.endswith(".ckpt") else checkpoint_name + ".ckpt"
        )

    def get_checkpoint(self) -> str:
        artifact_path = os.path.join(CHECKPOINT_PATH, self.checkpoint_name)

        local_path = self.client.download_artifacts(
            self.run_id,
            artifact_path,
            dst_path=".",
        )
        return local_path

import os
from abc import ABC, abstractmethod


class CheckpointProvider(ABC):
    @abstractmethod
    def get_checkpoint(self) -> str:
        """Loads the checkpoint data from the specified path."""
        pass


class MLFlowCheckpointProvider(CheckpointProvider):
    def __init__(
        self, tracking_uri: str, experiment_name: str, run_id: str, checkpoint_name: str
    ):
        from mlflow.tracking import MlflowClient

        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.checkpoint_name = checkpoint_name

    def get_checkpoint(self) -> str:
        artifact_path = os.path.join(
            self.experiment_name, self.run_id, "checkpoint", self.checkpoint_name
        )
        local_path = self.client.download_artifacts(
            self.experiment_name,
            artifact_path,
        )
        return local_path

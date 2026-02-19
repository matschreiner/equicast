import os

from mlflow.exceptions import MlflowException
from pytorch_lightning.loggers import MLFlowLogger as MLFlowLoggerParent

from equicast import CHECKPOINT_PATH


class MLFlowLogger(MLFlowLoggerParent):
    def log_hyperparams(self, params):
        try:
            super().log_hyperparams(params)
        except MlflowException:
            pass  # Ignore duplicate params when resuming a run

    def after_save_checkpoint(self, filepath: str):
        try:
            self.log_artifact(filepath, artifact_path=CHECKPOINT_PATH)
        except Exception as e:
            print(f"Warning: Could not upload checkpoint to MLflow: {e}")

    def log_artifact(self, local_path, artifact_path, *args, **kwargs):
        self.experiment.log_artifact(self.run_id, local_path, artifact_path=artifact_path, *args, **kwargs)

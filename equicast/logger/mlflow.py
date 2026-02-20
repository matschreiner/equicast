import os

import yaml
from mlflow.exceptions import MlflowException
from pytorch_lightning.loggers import MLFlowLogger as MLFlowLoggerParent

from equicast import CHECKPOINT_PATH, TRACKING_URI


def fix_artifact_location(experiment_name: str):
    """Rewrite meta.yaml artifact_location to the local tracking URI.

    Needed when an experiment was created on a different machine (e.g. Leonardo)
    and its meta.yaml still points to that machine's absolute path.
    """
    meta_path = os.path.join(TRACKING_URI, experiment_name, "meta.yaml")
    if not os.path.exists(meta_path):
        return
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    expected = f"{TRACKING_URI}/{experiment_name}"
    if meta.get("artifact_location") != expected:
        meta["artifact_location"] = expected
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)


class MLFlowLogger(MLFlowLoggerParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        fix_artifact_location(self._experiment_name)

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

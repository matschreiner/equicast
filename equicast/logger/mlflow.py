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


def resolve_run(run_name_or_id: str) -> str:
    """Resolve a run name or ID to a run ID."""
    import mlflow

    try:
        run = mlflow.get_run(run_name_or_id)
        return run.info.run_id
    except MlflowException:
        pass

    runs = mlflow.search_runs(
        filter_string=f"attributes.run_name = '{run_name_or_id}'",
        search_all_experiments=True,
    )
    if runs.empty:
        raise ValueError(f"No run found with name or id '{run_name_or_id}'")
    return runs.iloc[0]["run_id"]


class MLFlowLogger(MLFlowLoggerParent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rank == 0:
            fix_artifact_location(self._experiment_name)

    def log_hyperparams(self, params):
        first_call = not self._initialized
        try:
            super().log_hyperparams(params)
        except MlflowException:
            pass  # Ignore duplicate params when resuming a run
        if first_call and self.rank == 0:
            import mlflow
            run_name = mlflow.get_run(self.run_id).info.run_name
            line = f"MLflow run: {run_name}  |  id: {self.run_id}  |  experiment: {self._experiment_name}"
            print(f"\n{line}\n{'-' * len(line)}\n")

    def after_save_checkpoint(self, filepath: str):
        try:
            self.log_artifact(filepath, artifact_path=CHECKPOINT_PATH)
        except Exception as e:
            print(f"Warning: Could not upload checkpoint to MLflow: {e}")

    def log_artifact(self, local_path, artifact_path, *args, **kwargs):
        self.experiment.log_artifact(self.run_id, local_path, artifact_path=artifact_path, *args, **kwargs)

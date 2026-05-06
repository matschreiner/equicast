import os

import yaml
from mlflow.exceptions import MlflowException
from pytorch_lightning.loggers import MLFlowLogger as MLFlowLoggerParent


def fix_artifact_location(experiment_name: str, tracking_uri: str):
    """Rewrite meta.yaml artifact_location to the local tracking URI.

    Needed when an experiment was created on a different machine (e.g. Leonardo)
    and its meta.yaml still points to that machine's absolute path.
    """
    meta_path = os.path.join(tracking_uri, experiment_name, "meta.yaml")
    if not os.path.exists(meta_path):
        return
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    expected = f"{tracking_uri}/{experiment_name}"
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
        fix_artifact_location(self._experiment_name, self._tracking_uri)

    def log_hyperparams(self, params):
        try:
            super().log_hyperparams(params)
        except MlflowException:
            pass  # Ignore duplicate params when resuming a run

    def log_metrics(self, metrics, step=None, **kwargs):
        try:
            super().log_metrics(metrics, step=step, **kwargs)
        except MlflowException as e:
            print(f"Warning: MLflow log_metrics failed (step={step}): {e}")

    def log_config(self, cfg):
        import tempfile

        from omegaconf import OmegaConf

        self.log_hyperparams(dict(OmegaConf.to_container(cfg, resolve=True)))
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            OmegaConf.save(cfg, f)
            tmp_path = f.name
        try:
            self.experiment.log_artifact(self.run_id, tmp_path, artifact_path="config")
        finally:
            os.unlink(tmp_path)

    def after_save_checkpoint(self, filepath: str):
        try:
            self.experiment.log_artifact(self.run_id, filepath, artifact_path="checkpoints")
        except Exception as e:
            print(f"Warning: Could not log checkpoint to MLflow: {e}")

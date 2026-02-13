import os

from pytorch_lightning.loggers import MLFlowLogger as MLFlowLoggerParent

from equicast import CHECKPOINT_PATH


class MLFlowLogger(MLFlowLoggerParent):
    def after_save_checkpoint(self, checkpoint_callback):
        if checkpoint_callback.dirpath is None or checkpoint_callback.filename is None:
            return
        local_path = os.path.join(checkpoint_callback.dirpath, checkpoint_callback.filename + ".ckpt")
        try:
            self.log_artifact(local_path, artifact_path=CHECKPOINT_PATH)
        except Exception as e:
            print(f"Warning: Could not upload checkpoint to MLflow: {e}")

    def log_artifact(self, local_path, artifact_path, *args, **kwargs):
        self.experiment.log_artifact(self.run_id, local_path, artifact_path=artifact_path, *args, **kwargs)

import os

from pytorch_lightning.loggers import MLFlowLogger as MLFlowLoggerParent


class MLFlowLogger(MLFlowLoggerParent):
    def after_save_checkpoint(self, checkpoint_callback):
        local_path = os.path.join(
            checkpoint_callback.dirpath, checkpoint_callback.filename + ".ckpt"
        )
        self.log_artifact(local_path, artifact_path="checkpoints")

    def log_artifact(self, local_path, artifact_path, *args, **kwargs):
        self.experiment.log_artifact(
            self.run_id, local_path, artifact_path=artifact_path, *args, **kwargs
        )

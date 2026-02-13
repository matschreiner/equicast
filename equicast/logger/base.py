"""Base logger interface extending PyTorch Lightning's Logger."""

from abc import abstractmethod

from pytorch_lightning.loggers import Logger


class BaseLogger(Logger):
    """
    Extended PyTorch Lightning Logger with artifact logging support.

    Adds log_artifact method to the standard PyTorch Lightning logger interface.
    """

    @abstractmethod
    def log_artifact(self, local_path: str) -> None:
        """
        Log an artifact (file) for the experiment.

        Args:
            local_path: Path to the local file to log
        """
        pass

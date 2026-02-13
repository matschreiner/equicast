"""CSV logger with artifact support."""

import shutil
from pathlib import Path

from pytorch_lightning.loggers import CSVLogger as CSVLoggerParent

from equicast.logger.base import BaseLogger


class CSVLogger(CSVLoggerParent, BaseLogger):
    """
    CSV logger that extends PyTorch Lightning's CSVLogger with artifact logging.

    Logs metrics to CSV files and saves artifacts to an artifacts/ subdirectory.
    """

    def __init__(self, save_dir: str, name: str = "default", version: str | None = None):
        """
        Initialize CSV logger.

        Args:
            save_dir: Root directory for logs
            name: Experiment name
            version: Experiment version (optional, auto-generated if not provided)
        """
        super().__init__(save_dir=save_dir, name=name, version=version)

        # Create artifacts directory
        self.artifacts_dir = Path(self.log_dir) / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def log_artifact(self, local_path: str) -> None:
        """
        Copy an artifact to the artifacts directory.

        Args:
            local_path: Path to the local file to log
        """
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Artifact not found: {local_path}")

        dest = self.artifacts_dir / src.name

        # Copy the file
        shutil.copy2(src, dest)
        print(f"Logged artifact: {src.name} -> {dest}")

import logging
import os
import subprocess
import tempfile
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class CheckpointProvider(ABC):
    @abstractmethod
    def get_checkpoint(self) -> str:
        """Return a local path to the checkpoint file."""
        pass


class LocalCheckpointProvider(CheckpointProvider):
    """Load checkpoint from a local file path."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def get_checkpoint(self) -> str:
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        return self.checkpoint_path


class RemoteCheckpointProvider(CheckpointProvider):
    """Load checkpoint from a remote server via SCP or rsync."""

    def __init__(
        self,
        remote_path: str,
        host: str,
        local_cache_dir: str | None = None,
        method: str = "scp",
    ):
        """
        Args:
            remote_path: Path to checkpoint on remote server
            host: SSH host (e.g., "ohm", "user@ohm")
            local_cache_dir: Local directory to cache the checkpoint (default: temp dir)
            method: Transfer method, "scp" or "rsync"
        """
        if method not in ("scp", "rsync"):
            raise ValueError(f"method must be 'scp' or 'rsync', got {method!r}")
        self.remote_path = remote_path
        self.host = host
        self.local_cache_dir = local_cache_dir or tempfile.gettempdir()
        self.method = method
        self.local_path = None

    def get_checkpoint(self) -> str:
        if self.local_path and os.path.exists(self.local_path):
            logger.info(f"Using cached checkpoint: {self.local_path}")
            return self.local_path

        os.makedirs(self.local_cache_dir, exist_ok=True)
        checkpoint_name = os.path.basename(self.remote_path)
        self.local_path = os.path.join(self.local_cache_dir, checkpoint_name)
        remote_spec = f"{self.host}:{self.remote_path}"
        logger.info(f"Fetching checkpoint from {remote_spec} via {self.method}")
        logger.info(f"Destination: {self.local_path}")

        try:
            if self.method == "scp":
                subprocess.run(
                    ["scp", remote_spec, self.local_path],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            else:
                subprocess.run(
                    ["rsync", "-P", remote_spec, self.local_path],
                    check=True,
                )
            size_mb = os.path.getsize(self.local_path) / (1024 * 1024)
            logger.info(f"Checkpoint downloaded: {size_mb:.1f} MB")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to fetch checkpoint from {remote_spec}: {e}")

        return self.local_path


class MLFlowCheckpointProvider(CheckpointProvider):
    """Load checkpoint from an MLflow run."""

    def __init__(self, tracking_uri: str, run_id: str, checkpoint_name: str):
        from mlflow.tracking import MlflowClient

        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.run_id = run_id
        self.checkpoint_name = checkpoint_name if checkpoint_name.endswith(".ckpt") else checkpoint_name + ".ckpt"

    def get_checkpoint(self) -> str:
        artifact_path = self._find_artifact(self.checkpoint_name)
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = self.client.download_artifacts(self.run_id, artifact_path, dst_path=tmp_dir)
            # Copy out of the temp dir so it survives the context manager
            import shutil
            dest = os.path.join(tempfile.gettempdir(), self.checkpoint_name)
            shutil.copy2(local_path, dest)
        return dest

    def _find_artifact(self, name: str, path: str = "") -> str:
        for artifact in self.client.list_artifacts(self.run_id, path or None):
            if artifact.is_dir:
                try:
                    return self._find_artifact(name, artifact.path)
                except FileNotFoundError:
                    continue
            elif os.path.basename(artifact.path) == name:
                return artifact.path
        raise FileNotFoundError(f"Artifact '{name}' not found in run {self.run_id}")

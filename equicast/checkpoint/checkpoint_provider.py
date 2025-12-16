import os
import subprocess
import tempfile
from abc import ABC, abstractmethod

from equicast import CHECKPOINT_PATH


class CheckpointProvider(ABC):
    @abstractmethod
    def get_checkpoint(self) -> str:
        """Loads the checkpoint data from the specified path."""
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
    """Load checkpoint from a remote server via SSH/SCP."""

    def __init__(self, remote_path: str, host: str, local_cache_dir: str = None):
        """
        Args:
            remote_path: Path to checkpoint on remote server
            host: SSH host (e.g., "ohm", "user@ohm")
            local_cache_dir: Local directory to cache the checkpoint (default: temp dir)
        """
        self.remote_path = remote_path
        self.host = host
        self.local_cache_dir = local_cache_dir or tempfile.gettempdir()
        self.local_path = None

    def get_checkpoint(self) -> str:
        if self.local_path and os.path.exists(self.local_path):
            print(f"Using cached checkpoint: {self.local_path}")
            return self.local_path

        # Create cache directory if needed
        os.makedirs(self.local_cache_dir, exist_ok=True)

        # Generate local filename from remote path
        checkpoint_name = os.path.basename(self.remote_path)
        self.local_path = os.path.join(self.local_cache_dir, checkpoint_name)

        # Copy from remote server using scp
        remote_spec = f"{self.host}:{self.remote_path}"
        print(f"Copying checkpoint from {remote_spec}")

        try:
            subprocess.run(
                ["scp", remote_spec, self.local_path],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"Checkpoint copied to {self.local_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to copy checkpoint from {remote_spec}: {e.stderr}"
            )

        return self.local_path


class MLFlowCheckpointProvider(CheckpointProvider):
    def __init__(self, tracking_uri: str, run_id: str, checkpoint_name: str):
        from mlflow.tracking import MlflowClient

        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.run_id = run_id

        self.checkpoint_name = (
            checkpoint_name if checkpoint_name.endswith(".ckpt") else checkpoint_name + ".ckpt"
        )

    def get_checkpoint(self) -> str:
        artifact_path = os.path.join(CHECKPOINT_PATH, self.checkpoint_name)

        local_path = self.client.download_artifacts(
            self.run_id,
            artifact_path,
            dst_path=".",
        )
        return local_path

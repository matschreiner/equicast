"""Utility functions for loading models from MLflow."""

import os
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient

from equicast.checkpoint.checkpoint_provider import MLFlowCheckpointProvider
from equicast.model import Model


def get_mlflow_config(config_path: str | None = None) -> dict:
    """
    Load MLflow configuration from YAML file.

    Args:
        config_path: Path to mlflow_config.yaml. If None, looks in config/ directory.

    Returns:
        Dictionary with mlflow configuration
    """
    if config_path is None:
        # Default to config/mlflow_config.yaml in project root
        config_path = Path(__file__).parent.parent.parent / "config" / "mlflow_config.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config["mlflow"]


def get_run_info(run_id: str, tracking_uri: str | None = None):
    """
    Get run information from MLflow.

    Args:
        run_id: MLflow run ID
        tracking_uri: MLflow tracking URI. If None, loads from config.

    Returns:
        MLflow Run object with metadata
    """
    if tracking_uri is None:
        mlflow_config = get_mlflow_config()
        tracking_uri = mlflow_config["tracking_uri"]

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    return run


def load_model_from_mlflow(
    run_id: str,
    checkpoint_name: str | None = None,
    tracking_uri: str | None = None,
    map_location: str | None = None,
) -> Model:
    """
    Load a model from MLflow using only the run ID.

    This function automatically:
    1. Fetches the tracking URI from config if not provided
    2. Downloads the checkpoint from MLflow artifacts
    3. Loads the model from the checkpoint

    Args:
        run_id: MLflow run ID (e.g., "abc123def456")
        checkpoint_name: Name of checkpoint in artifacts (without .ckpt extension).
                        If None, uses default from config.
        tracking_uri: MLflow tracking URI. If None, loads from config.
        map_location: Device to load model on ('cuda', 'cpu', etc.)

    Returns:
        Loaded Model instance

    Examples:
        >>> # Load latest model from run
        >>> model = load_model_from_mlflow("abc123def456")

        >>> # Load specific checkpoint
        >>> model = load_model_from_mlflow("abc123def456", checkpoint_name="epoch_10")

        >>> # Load to specific device
        >>> model = load_model_from_mlflow("abc123def456", map_location="cuda:1")

        >>> # Use custom MLflow server
        >>> model = load_model_from_mlflow(
        ...     "abc123def456",
        ...     tracking_uri="http://mlflow.example.com"
        ... )
    """
    # Load config if parameters not provided
    mlflow_config = get_mlflow_config()

    if tracking_uri is None:
        tracking_uri = mlflow_config["tracking_uri"]

    if checkpoint_name is None:
        checkpoint_name = mlflow_config.get("default_checkpoint_name", "best_model")

    # Get run info for logging
    run = get_run_info(run_id, tracking_uri)
    experiment_id = run.info.experiment_id
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment(experiment_id)

    print(f"Loading model from MLflow:")
    print(f"  Experiment: {experiment.name}")
    print(f"  Run ID: {run_id}")
    print(f"  Run Name: {run.info.run_name}")
    print(f"  Checkpoint: {checkpoint_name}")

    # Download checkpoint from MLflow
    checkpoint_provider = MLFlowCheckpointProvider(
        tracking_uri=tracking_uri,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
    )

    checkpoint_path = checkpoint_provider.get_checkpoint()
    print(f"  Downloaded to: {checkpoint_path}")

    # Load model from checkpoint
    model = Model.load_from_checkpoint(checkpoint_path, map_location=map_location)

    return model


def load_checkpoint_path_from_mlflow(
    run_id: str,
    checkpoint_name: str | None = None,
    tracking_uri: str | None = None,
) -> str:
    """
    Download checkpoint from MLflow and return the local path.

    Useful if you need the checkpoint path for custom loading logic.

    Args:
        run_id: MLflow run ID
        checkpoint_name: Name of checkpoint in artifacts
        tracking_uri: MLflow tracking URI. If None, loads from config.

    Returns:
        Local path to downloaded checkpoint file

    Examples:
        >>> # Get checkpoint path
        >>> ckpt_path = load_checkpoint_path_from_mlflow("abc123def456")
        >>> # Use with custom loading
        >>> model = Model.load_from_checkpoint(ckpt_path, strict=False)
    """
    # Load config if parameters not provided
    mlflow_config = get_mlflow_config()

    if tracking_uri is None:
        tracking_uri = mlflow_config["tracking_uri"]

    if checkpoint_name is None:
        checkpoint_name = mlflow_config.get("default_checkpoint_name", "best_model")

    # Download checkpoint
    checkpoint_provider = MLFlowCheckpointProvider(
        tracking_uri=tracking_uri,
        run_id=run_id,
        checkpoint_name=checkpoint_name,
    )

    return checkpoint_provider.get_checkpoint()


def list_checkpoints_in_run(run_id: str, tracking_uri: str | None = None) -> list[str]:
    """
    List all available checkpoints in an MLflow run.

    Args:
        run_id: MLflow run ID
        tracking_uri: MLflow tracking URI. If None, loads from config.

    Returns:
        List of checkpoint names available in the run

    Examples:
        >>> # See what checkpoints are available
        >>> checkpoints = list_checkpoints_in_run("abc123def456")
        >>> print(checkpoints)
        ['best_model', 'epoch_10', 'epoch_20', 'last']
    """
    if tracking_uri is None:
        mlflow_config = get_mlflow_config()
        tracking_uri = mlflow_config["tracking_uri"]

    client = MlflowClient(tracking_uri=tracking_uri)

    # List artifacts in the checkpoints directory
    from equicast import CHECKPOINT_PATH

    artifacts = client.list_artifacts(run_id, path=CHECKPOINT_PATH)

    # Extract checkpoint names (without .ckpt extension)
    checkpoint_names = [
        os.path.splitext(artifact.path.split("/")[-1])[0] for artifact in artifacts if artifact.path.endswith(".ckpt")
    ]

    return checkpoint_names

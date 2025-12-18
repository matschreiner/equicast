"""Metrics tracking for model evaluation."""

from abc import ABC, abstractmethod

import torch

from equicast.data.data_handler import BaseDataHandler


class BaseMetricsTracker(ABC):
    """Abstract base class for metrics tracking."""

    @abstractmethod
    def compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute various metrics on predictions and targets.

        Args:
            pred: Model predictions
            target: Target values

        Returns:
            Dictionary of metric names to values
        """
        pass


class MetricsTracker(BaseMetricsTracker):
    """Tracks various statistics on model predictions and targets."""

    def __init__(self, data_handler: BaseDataHandler):
        """
        Initialize metrics tracker.

        Args:
            data_handler: DataHandler to get variable names and indices
        """
        self.data_handler = data_handler
        # Create mapping from output index position to variable name
        self.index_to_name = {
            v: k for k, v in data_handler.name_to_index.items()
        }
        self.out_idxs = data_handler.out_idxs

    def compute_metrics(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Compute various metrics on predictions and targets.

        Args:
            pred: Model predictions with shape [..., num_output_features]
            target: Target values with shape [..., num_output_features]

        Returns:
            Dictionary of metric names to values
        """
        # Compute errors for all features at once
        error = pred - target

        # Compute bias across all dimensions except the last (features)
        # This reduces to shape [num_output_features]
        dims = tuple(range(error.ndim - 1))
        bias = error.mean(dim=dims)

        # Compute unbiased variance (subtract bias before computing variance)
        unbiased_error = error - bias
        variance = unbiased_error.var(dim=dims)

        # Create metrics dict with proper variable names
        metrics = {}
        for i, idx in enumerate(self.out_idxs):
            var_name = self.index_to_name[idx]
            metrics[f"bias/{var_name}"] = bias[i]
            metrics[f"variance/{var_name}"] = variance[i]

        return metrics

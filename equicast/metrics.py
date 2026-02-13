"""Metrics tracking for model evaluation."""

from abc import ABC, abstractmethod

import torch

from equicast.data.data_handler import BaseDataHandler
from equicast.data.equivariant_data_handler import EquivariantGraphDataHandler


class BaseMetricsTracker(ABC):
    """Abstract base class for metrics tracking."""

    @abstractmethod
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
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
        self.index_to_name = {v: k for k, v in data_handler.name_to_index.items()}
        self.out_idxs = data_handler.out_idxs

    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute various metrics on predictions and targets.

        Args:
            pred: Model predictions with shape [..., num_output_features]
            target: Target values with shape [..., num_output_features]

        Returns:
            Dictionary of metric names to values
        """
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
            metrics[f"metrics/{var_name}_bias"] = bias[i]
            metrics[f"metrics/{var_name}_variance"] = variance[i]

        return metrics


class EquivariantMetricsTracker(BaseMetricsTracker):
    """Tracks metrics for equivariant models with scalar and vector outputs."""

    def __init__(self, data_handler: EquivariantGraphDataHandler):
        self.data_handler = data_handler
        self.index_to_name = {v: k for k, v in data_handler.name_to_index.items()}
        self.out_idxs = data_handler.out_idxs
        self.vector_names = list(data_handler.feature_config.prognostic_vector.keys())

    def compute_metrics(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        metrics = {}

        # Scalar metrics — same logic as MetricsTracker
        scalar_error = pred["scalar"] - target["scalar"]
        dims = tuple(range(scalar_error.ndim - 1))
        scalar_bias = scalar_error.mean(dim=dims)
        unbiased_error = scalar_error - scalar_bias
        scalar_variance = unbiased_error.var(dim=dims)

        for i, idx in enumerate(self.out_idxs):
            var_name = self.index_to_name[idx]
            metrics[f"metrics/{var_name}_bias"] = scalar_bias[i]
            metrics[f"metrics/{var_name}_variance"] = scalar_variance[i]

        # Vector metrics — pred/target shape: [nodes, num_vectors, 2]
        vector_error = pred["vector"] - target["vector"]
        # Reduce over all dims except vector-feature dim (1) and xy dim (2)
        reduce_dims = tuple(
            d
            for d in range(vector_error.ndim)
            if d not in (vector_error.ndim - 2, vector_error.ndim - 1)
        )
        vector_bias = vector_error.mean(dim=reduce_dims)  # [num_vectors, 2]
        unbiased_vector_error = vector_error - vector_bias
        vector_variance = unbiased_vector_error.var(dim=reduce_dims)

        for i, name in enumerate(self.vector_names):
            metrics[f"metrics/{name}_u_bias"] = vector_bias[i, 0]
            metrics[f"metrics/{name}_v_bias"] = vector_bias[i, 1]
            metrics[f"metrics/{name}_u_variance"] = vector_variance[i, 0]
            metrics[f"metrics/{name}_v_variance"] = vector_variance[i, 1]

        return metrics

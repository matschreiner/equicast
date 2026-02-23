"""Metrics tracking for model evaluation."""

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseMetricsTracker(ABC):
    """Abstract base class for metrics tracking."""

    @abstractmethod
    def compute_metrics(self, pred: Any, target: Any) -> dict[str, torch.Tensor]:
        """
        Compute various metrics on predictions and targets.

        Args:
            pred: Model predictions
            target: Target values

        Returns:
            Dictionary of metric names to values
        """
        pass


class WeatherBenchTracker(BaseMetricsTracker):
    """Tracks RMSE for standard WeatherBench variables: z_500, t_850, q_700, wind_850."""

    SCALAR_VARIABLES = ["z_500", "t_850", "q_700"]
    WIND_COMPONENTS = [("u_850", "v_850")]

    def __init__(self, data_handler):
        self.data_handler = data_handler
        n2i = data_handler.name_to_index
        self.scalar_idxs = {name: n2i[name] for name in self.SCALAR_VARIABLES}
        self.wind_idxs = {f"wind_{u.split('_')[1]}": (n2i[u], n2i[v]) for u, v in self.WIND_COMPONENTS}

    def compute_metrics(self, pred, target) -> dict[str, torch.Tensor]:
        pred_raw = self.data_handler.to_raw(pred)
        target_raw = self.data_handler.to_raw(target)

        metrics = {}
        for name, idx in self.scalar_idxs.items():
            rmse = (pred_raw[:, idx] - target_raw[:, idx]).pow(2).mean().sqrt()
            metrics[f"metrics/{name}_rmse"] = rmse

        for name, (u_idx, v_idx) in self.wind_idxs.items():
            u_err = pred_raw[:, u_idx] - target_raw[:, u_idx]
            v_err = pred_raw[:, v_idx] - target_raw[:, v_idx]
            metrics[f"metrics/{name}_rmse"] = (u_err**2 + v_err**2).mean().sqrt()

        return metrics

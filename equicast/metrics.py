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
    """Metrics tracker for WeatherBench dataset."""

    def compute_metrics(
        self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        __import__("pdb").set_trace()  # TODO delme

        return {"train/rmse_u": torch.Tensor(0.0)}

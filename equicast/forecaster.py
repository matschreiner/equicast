import os

import torch
from tqdm import tqdm

from equicast.logger import BaseLogger
from equicast.model.model import Model
from equicast.visualization import make_comparison_video


class Forecaster:
    """
    Forecaster that delegates all preprocessing to the Model.

    The model handles scaling and feature routing internally, so the
    forecaster just manages the autoregressive loop.
    """

    def __init__(self, model: Model, logger: BaseLogger | None = None):
        self.model = model
        self.logger = logger

    def forecast(self, timeseries, steps=-1, output_dir="."):
        """
        Autoregressively forecast for a given number of steps.

        Args:
            timeseries: Tensor of shape (steps + 1, num_nodes, num_features)
            steps: Number of steps to forecast
            output_dir: Directory where forecast outputs will be saved

        Returns:
            List of predictions (model handles scaling internally)
        """

        predictions = []
        current_state = timeseries[0]
        for step in range(10):
            pred = self.model(current_state)
            predictions.append(pred)

            next_state = timeseries[step + 1]
            current_state = self.model.data_handler.update_state_with_prediction(
                next_state,
                pred,
            )

        return predictions

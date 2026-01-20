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

    def forecast(self, timeseries, steps=-1, output_dir=".", feature_idx=0):
        """
        Autoregressively forecast for a given number of steps.

        Args:
            timeseries: List of states for each timestep
            steps: Number of steps to forecast (-1 for all available)
            output_dir: Directory where forecast outputs will be saved
            feature_idx: Feature index to visualize (default: 0)

        Returns:
            List of predictions (model handles scaling internally)
        """
        predictions = []
        condition = timeseries[0]
        num_steps = steps if steps > 0 else len(timeseries) - 1

        with torch.no_grad():
            for step in tqdm(range(num_steps), desc="Forecasting"):
                condition, pred = self.model.step_forward(
                    condition,
                    timeseries[step + 1],
                )
                predictions.append(pred.detach())

        if self.logger is not None:
            self._save_visualization(
                predictions, timeseries, num_steps, output_dir, feature_idx
            )

        return predictions

    def _save_visualization(
        self, predictions, timeseries, num_steps, output_dir, feature_idx
    ):
        """Save comparison video of predictions vs ground truth."""
        preds = torch.stack(predictions, dim=0).squeeze()
        ground_truth = torch.stack(
            [
                self.model.data_handler.get_output_features(graph["grid"].raw_input)
                for graph in timeseries[1 : num_steps + 1]
            ]
        )

        video_path = os.path.join(output_dir, "forecast_comparison.mp4")
        make_comparison_video(
            predictions=preds.cpu().numpy()[..., feature_idx],
            targets=ground_truth.squeeze().cpu().numpy()[..., feature_idx],
            latlon=timeseries[0]["grid"].x.cpu().numpy(),
            output_path=video_path,
        )

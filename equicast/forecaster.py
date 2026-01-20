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
        input = timeseries[0]["input"]
        num_steps = steps if steps > 0 else len(timeseries) - 1
        ground_truths = [t["target"] for t in timeseries[:num_steps]]
        feature_idx = 1

        with torch.no_grad():
            for target in tqdm(ground_truths, desc="Forecasting"):
                input, prediction = self.model.step_forward(
                    input,
                    target,
                )

                predictions.append(prediction)

        if self.logger is not None:
            self._save_visualization(
                predictions, ground_truths, output_dir, feature_idx
            )

        return predictions

    def _save_visualization(
        self, predictions, ground_truths, output_dir, feature_idx
    ):
        """Save comparison video of predictions vs ground truth."""
        nodes = self.model.nodes

        pred = [p[nodes].data for p in predictions]
        pred = torch.stack(pred, dim=0)
        pred = self.model.data_handler.get_output_features(pred)

        ground_truth = [g[nodes].data for g in ground_truths[: len(predictions)]]
        ground_truth = torch.stack(ground_truth, dim=0)
        ground_truth = self.model.data_handler.get_output_features(ground_truth)

        video_path = os.path.join(output_dir, "forecast_comparison.mp4")
        make_comparison_video(
            predictions=pred.cpu().numpy()[..., feature_idx],
            targets=ground_truth.cpu().numpy()[..., feature_idx],
            latlon=ground_truths[0][nodes].x.cpu().numpy(),
            output_path=video_path,
        )

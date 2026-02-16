import os

import torch
import xarray as xr
from tqdm import tqdm

from equicast.logger import BaseLogger
from equicast.model.model import Model


class Forecaster:
    """
    Forecaster that delegates all preprocessing to the Model.

    The model handles scaling and feature routing internally, so the
    forecaster just manages the autoregressive loop.
    """

    def __init__(self, model: Model, logger: BaseLogger | None = None):
        self.model = model
        self.logger = logger

    def forecast(self, timeseries, output_dir="test_output"):
        """
        Autoregressively forecast over the given timeseries.

        Args:
            timeseries: List of graph states (consecutive timesteps)
            output_dir: Directory where forecast outputs will be saved

        Returns:
            List of predictions (model handles scaling internally)
        """
        input_ = timeseries[0]
        predictions = []
        ground_truth = [frame.clone() for frame in timeseries[1:]]

        with torch.no_grad():
            for next_ in tqdm(timeseries[1:], desc="Forecasting"):
                input_, prediction = self.model.step_forward(
                    input_,
                    next_,
                )

                predictions.append(prediction)

        self._save_zarr(predictions, output_dir, "predictions.zarr")
        self._save_zarr(ground_truth, output_dir, "input_timeseries.zarr")

        return predictions

    def _save_zarr(self, timeseries, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        ds = xr.concat([self.model.data_handler.to_cf(frame) for frame in timeseries], dim="time")  # type: ignore
        ds.to_zarr(path, mode="w")

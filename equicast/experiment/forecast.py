import os
from dataclasses import dataclass
from typing import Any

from equicast.data.data_handler import BaseDataHandler
from equicast.experiment.config import ExperimentConfig
from equicast.forecaster import Forecaster
@dataclass
class ForecastConfig(ExperimentConfig):
    forecaster: Forecaster
    input_timeseries: list[Any]
    target_timeseries: list[Any]
    data_handler: BaseDataHandler
    model_id: str = ""
    experiment_name: str = "forecast"
    output_dir: str = "forecasts"

    def run(self):
        output_dir = f"{self.output_dir}/{self.model_id}"
        forecast = self.forecaster.forecast(timeseries=self.input_timeseries)

        forecast_path = os.path.join(output_dir, "forecast.zarr")
        target_path = os.path.join(output_dir, "target_forecast.zarr")
        initial_path = os.path.join(output_dir, "initial_state.zarr")

        self.data_handler.outputs_to_zarr(forecast, forecast_path)
        self.data_handler.outputs_to_zarr(self.target_timeseries, target_path)
        self.data_handler.outputs_to_zarr([self.input_timeseries[0]], initial_path)

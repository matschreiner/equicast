import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from equicast.data.data_handler import BaseDataHandler
from equicast.experiment.config import ExperimentConfig
from equicast.forecaster import Forecaster
@dataclass
class ForecastConfig(ExperimentConfig):
    forecaster: Forecaster
    input_timeseries: list[Any]
    data_handler: BaseDataHandler
    model_id: str = ""
    experiment_name: str = "forecast"
    output_dir: str = "forecasts"
    meta: dict = field(default_factory=dict)

    def run(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")
        output_dir = f"{self.output_dir}/{self.model_id}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        forecast = self.forecaster.forecast(timeseries=self.input_timeseries)

        self.data_handler.outputs_to_zarr(forecast, os.path.join(output_dir, "forecast.zarr"))
        self.data_handler.outputs_to_zarr([self.input_timeseries[0]], os.path.join(output_dir, "initial_state.zarr"))

        meta = {**self.meta, "created_at": timestamp}
        with open(os.path.join(output_dir, "forecast.json"), "w") as f:
            json.dump(meta, f, indent=2)

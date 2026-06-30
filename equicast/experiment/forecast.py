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
    output_name: str = ""
    meta: dict = field(default_factory=dict)

    def run(self):
        name = self.output_name or datetime.now().strftime("%Y-%m-%d_%H:%M")
        output_dir = f"{self.output_dir}/{self.model_id}/{name}"
        forecasts_dir = os.path.join(output_dir, "forecasts")
        os.makedirs(forecasts_dir, exist_ok=True)

        start_idx = self.meta.get("start_idx", 0)
        prefix = f"forecast_{start_idx:04d}" if self.meta.get("batch") else "forecast"

        forecast = self.forecaster.forecast(timeseries=self.input_timeseries)
        self.data_handler.outputs_to_zarr(forecast, os.path.join(forecasts_dir, f"{prefix}.zarr"))

        meta = {**self.meta, "output_dir": output_dir}
        with open(os.path.join(forecasts_dir, f"{prefix}.json"), "w") as f:
            json.dump(meta, f, indent=2)

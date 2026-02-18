import os
from dataclasses import dataclass
from typing import Any

from equicast.data.data_handler import BaseDataHandler
from equicast.experiment.config import ExperimentConfig
from equicast.forecaster import Forecaster
from equicast.logger import BaseLogger


@dataclass
class ForecastConfig(ExperimentConfig):
    forecaster: Forecaster
    input_timeseries: list[Any]
    target_timeseries: list[Any]
    logger: BaseLogger
    data_handler: BaseDataHandler
    model_id: str = ""
    experiment_name: str = "forecast"

    def run(self):
        output_dir = f"forecasts/{self.model_id}"
        predictions = self.forecaster.forecast(timeseries=self.input_timeseries)

        self.data_handler.save_outputs_to_cf(predictions, path=os.path.join(output_dir, "predictions"))
        self.data_handler.save_outputs_to_cf(self.target_timeseries, path=os.path.join(output_dir, "ground_truth"))

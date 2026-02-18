import torch
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

    def forecast(self, timeseries):
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

        with torch.no_grad():
            for next_ in tqdm(timeseries[1:], desc="Forecasting"):
                input_, prediction = self.model.step_forward(
                    input_,
                    next_,
                )

                predictions.append(prediction)

        return predictions

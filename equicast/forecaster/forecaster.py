import torch
from tqdm import tqdm

from equicast.logger import BaseLogger
from equicast.model.base import BaseModel


class Forecaster:
    """
    Forecaster that delegates all preprocessing to the Model.

    The model handles scaling and feature routing internally, so the
    forecaster just manages the autoregressive loop.
    """

    def __init__(self, model: BaseModel, logger: BaseLogger | None = None):
        self.model = model
        self.logger = logger

    def forecast(self, timeseries):
        """
        Autoregressively forecast over the given timeseries.

        Args:
            timeseries: List of individual graph states (one per timestep)

        Returns:
            List of predictions
        """
        device = next(self.model.parameters()).device
        n_input = getattr(self.model.data_handler, "n_input_frames", 1)
        window = timeseries[:n_input] if n_input > 1 else timeseries[0]
        forecast = []

        def to_device(x):
            return [g.to(device) for g in x] if isinstance(x, list) else x.to(device)

        with torch.no_grad():
            for next_ in tqdm(timeseries[n_input:], desc="Forecasting"):
                next_state, prediction = self.model.step_forward(to_device(window), next_.to(device))
                forecast.append(prediction.cpu())
                next_state = next_state.cpu()
                window = window[1:] + [next_state] if n_input > 1 else next_state

        return forecast

import torch

from equicast.model.model import Model


class Forecaster:
    """
    Simplified forecaster that delegates all preprocessing to the Model.

    The model handles scaling and feature routing internally, so the
    forecaster just manages the autoregressive loop.
    """

    def __init__(self, model: Model):
        self.model = model

    def forecast(self, time_series, graph, steps):
        """
        Autoregressively forecast for a given number of steps.

        Args:
            initial_state: Graph with raw initial conditions
            steps: Number of forecast steps
            forcing_sequence: Optional tensor of forcing variables for each step

        Returns:
            List of predictions (model handles scaling internally)
        """
        self.model.eval()
        predictions = []
        current_state = time_series[0]

        with torch.no_grad():
            for step in range(steps):
                pred = self.model(current_state)
                predictions.append(pred)
                current_state = (
                    self.model.data_handler.update_state_with_prediction(
                        time_series[step + 1],
                        pred,
                    )
                )

        return torch.stack(predictions, dim=0)  # [time, batch, nodes, features]

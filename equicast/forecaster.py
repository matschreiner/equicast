import torch

from equicast.model.model import Model


class Forecaster:
    """
    Forecaster that delegates all preprocessing to the Model.

    The model handles scaling and feature routing internally, so the
    forecaster just manages the autoregressive loop.
    """

    def __init__(self, model: Model):
        self.model = model

    def forecast(self, timeseries, graph, steps):
        """
        Autoregressively forecast for a given number of steps.

        Args:
            timeseries: Tensor of shape (steps + 1, num_nodes, num_features)
            graph: Graph data structure
            steps: Number of steps to forecast

        Returns:
            List of predictions (model handles scaling internally)
        """
        self.model.eval()
        predictions = []
        current_state = timeseries[0]

        with torch.no_grad():
            for step in range(steps):
                _graph = graph.clone()
                _graph["grid"].input_state = current_state

                pred = self.model(_graph)
                predictions.append(pred)
                current_state = (
                    self.model.data_handler.update_state_with_prediction(
                        timeseries[step + 1],
                        pred,
                    )
                )

        preds = torch.stack(predictions, dim=0)
        __import__("pdb").set_trace()  # TODO delme
        return preds

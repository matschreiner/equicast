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

    def forecast(self, timeseries, graph, steps=-1):
        """
        Autoregressively forecast for a given number of steps.

        Args:
            timeseries: Tensor of shape (steps + 1, num_nodes, num_features)
            graph: Graph data structure
            steps: Number of steps to forecast

        Returns:
            List of predictions (model handles scaling internally)
        """
        if steps == -1:
            steps = len(timeseries) - 1

        # Ensure timeseries and graph are on the same device as model
        device = next(self.model.parameters()).device
        timeseries = timeseries.to(device)
        graph = graph.to(device)

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

        field = 1

        import matplotlib.pyplot as plt

        from equicast.utils.vis import make_comparison_video

        preds = torch.stack(predictions)
        targets = self.model.data_handler.prepare_model_target(timeseries)
        fields = preds[:, :, field]
        print("prediction done")

        v = make_comparison_video(
            fields.cpu().numpy(),
            targets[:, :, field].cpu().numpy(),
            _graph["grid"].x.cpu().numpy(),
            title="Forecasted field",
            output_path="forecast.mp4",
            fps=1,
        )

        return preds

import torch
from tqdm import tqdm

from equicast.model.model import Model


class Forecaster:
    """
    Forecaster that delegates all preprocessing to the Model.

    The model handles scaling and feature routing internally, so the
    forecaster just manages the autoregressive loop.
    """

    def __init__(self, model: Model):
        self.model = model

    def forecast(self, timeseries, graph, steps=-1, output_dir="."):
        """
        Autoregressively forecast for a given number of steps.

        Args:
            timeseries: Tensor of shape (steps + 1, num_nodes, num_features)
            graph: Graph data structure
            steps: Number of steps to forecast
            output_dir: Directory where forecast outputs will be saved

        Returns:
            List of predictions (model handles scaling internally)
        """
        if steps == -1:
            steps = len(timeseries) - 1

        # Ensure timeseries and graph are on the same device as model

        device = next(self.model.parameters()).device
        timeseries = timeseries.to(device)
        graph = graph.to(device)
        predictions = []

        self.model.eval()
        current_state = timeseries[0].unsqueeze(0)

        with torch.no_grad():
            for step in tqdm(range(steps), desc="Forecasting"):
                _graph = graph.clone()
                _graph["grid"].data = current_state

                prediction = self.model(_graph)
                predictions.append(prediction)
                current_state = (
                    self.model.data_handler.update_state_with_prediction(
                        timeseries[step + 1].unsqueeze(0),
                        prediction,
                    )
                )

        field = 1

        import matplotlib.pyplot as plt
        from pathlib import Path

        from equicast.visualization import make_comparison_video

        preds = torch.stack(predictions)

        graph["grid"]
        timeseries = self.model.data_handler.scaler.transform(timeseries)
        targets = self.model.data_handler.get_output_features(timeseries)
        fields = preds[:, :, field]

        output_path = Path(output_dir) / "forecast.mp4"
        v = make_comparison_video(
            fields.cpu().numpy(),
            targets[:, :, field].cpu().numpy(),
            graph["grid"].x.cpu().numpy(),
            title="Forecasted field",
            output_path=str(output_path),
            fps=1,
        )

        return preds

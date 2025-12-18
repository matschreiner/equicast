import os

import torch
from tqdm import tqdm

from equicast.logger import BaseLogger
from equicast.model.model import Model
from equicast.visualization import make_comparison_video


class Forecaster:
    """
    Forecaster that delegates all preprocessing to the Model.

    The model handles scaling and feature routing internally, so the
    forecaster just manages the autoregressive loop.
    """

    def __init__(self, model: Model, logger: BaseLogger | None = None):
        self.model = model
        self.logger = logger

    def forecast(self, timeseries, steps=-1, output_dir="."):
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

        predictions = []
        current_state = timeseries[0]
        for step in range(10):
            pred = self.model(current_state)
            predictions.append(pred)

            next_state = timeseries[step + 1]
            current_state = self.model.data_handler.update_state_with_prediction(
                next_state,
                pred,
            )

        return predictions





            __import__("pdb").set_trace()  # TODO delme
            pred = self.model(current_state)

            __import__("pdb").set_trace()  # TODO delme

        __import__("pdb").set_trace()  # TODO delme
        #  graph["grid"].data = timeseries
        #  graph = self.model.data_handler.prepare_input(graph)
        #  self.model.eval()
        #
        #  with torch.no_grad():
        #      for step in tqdm(range(steps), desc="Forecasting"):
        #          __import__("pdb").set_trace()
        #          pred = model.predict(graph)
        #
        #
        #
        #
        #
        #
        #
        #
        #  if steps == -1:
        #      steps = len(timeseries) - 1
        #
        #  steps = 2
        #  device = next(self.model.parameters()).device
        #  timeseries = timeseries.to(device)
        #  graph = graph.to(device)
        #
        #  predictions = []
        #
        #  self.model.eval()
        #  #  current_state = timeseries[0].unsqueeze(0)
        #
        #  self.model.data_handler.get_input_features(timeseries)
        #  with torch.no_grad():
        #      for step in tqdm(range(steps), desc="Forecasting"):
        #          _graph = graph.clone()
        #          _graph["grid"].data = current_state
        #
        #          prediction = self.model(_graph)
        #          predictions.append(prediction)
        #          current_state = (
        #              self.model.data_handler.update_state_with_prediction(
        #                  timeseries[step + 1].unsqueeze(0),
        #                  prediction,
        #              )
        #          )
        #
        #  field = 1
        #  preds = torch.stack(predictions)
        #  preds_path = os.path.join(output_dir, "predictions.pt")
        #  torch.save(preds, preds_path)
        #  timeseries = self.model.data_handler.normalizer.transform(timeseries)
        #  targets = self.model.data_handler.get_output_features(timeseries)
        #  fields = preds[:, :, field]
        #  mp4_path = os.path.join(output_dir, "forecast.mp4")
        #  make_comparison_video(
        #      fields.cpu().numpy(),
        #      targets[:, :, field].cpu().numpy(),
        #      graph["grid"].x.cpu().numpy(),
        #      title="Forecasted field",
        #      output_path=str(mp4_path),
        #      fps=1,
        #  )
        #  if self.logger:
        #      self.logger.log_artifact(mp4_path)
        #      self.logger.log_artifact(preds_path)

        return preds

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

    def forecast(self, initial_state, steps, forcing_sequence=None):
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

        current_state = initial_state

        with torch.no_grad():
            for step in range(steps):
                # Model handles all preprocessing internally
                pred = self.model(current_state)
                predictions.append(pred)

                # Prepare next state for autoregressive loop
                current_state = self._prepare_next_state(
                    current_state,
                    pred,
                    (
                        forcing_sequence[step]
                        if forcing_sequence is not None
                        else None
                    ),
                )

        return torch.stack(predictions, dim=0)  # [time, batch, nodes, features]

    def _prepare_next_state(self, current_graph, prediction, forcing=None):
        """
        Prepare the next state from model prediction.

        Delegates to DataHandler for extracting prognostic variables and
        reconstructing the full state.

        Args:
            current_graph: Current graph (used to clone structure)
            prediction: Model output [prognostic, diagnostic] in scaled space
            forcing: Forcing variables for next timestep in physical space, optional

        Returns:
            Graph ready for next model forward pass
        """
        data_handler = self.model.data_handler

        # Extract prognostic variables (unscaled to physical space)
        prognostic = data_handler.extract_prognostic(prediction)

        # Reconstruct full feature vector
        next_state = data_handler.reconstruct_state(prognostic, forcing)

        # Clone graph structure and update input_state
        next_graph = current_graph.clone()
        next_graph["grid"].input_state = next_state

        return next_graph

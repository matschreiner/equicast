import torch

from equicast.data.data_handler import DataHandler


class Forecaster:
    def __init__(
        self,
        model: torch.nn.Module,
        data_handler: DataHandler,
    ):
        self.model = model
        self.data_handler = data_handler

    def forecast(self, initial_state, steps, forcing_sequence=None):
        """
        Autoregressively forecast for a given number of steps.

        Args:
            initial_state: Graph with initial conditions (scaled)
            steps: Number of forecast steps
            forcing_sequence: Optional tensor of forcing variables for each step
                            Shape: (steps, num_nodes, num_forcing_vars)

        Returns:
            List of predictions (unscaled, in physical units)
        """
        self.model.eval()
        predictions = []

        # Start with initial state
        current_state = initial_state

        with torch.no_grad():
            for step in range(steps):
                # Get model prediction (in scaled space)
                pred = self.model(current_state)

                # Inverse scale prediction to physical units for output
                pred_unscaled = self.data_handler.scaler.inverse_transform(pred)
                predictions.append(pred_unscaled)

                # Prepare next state for autoregressive loop
                current_state = self._prepare_next_state(
                    pred,
                    forcing_sequence[step] if forcing_sequence is not None else None
                )

        return predictions

    def _prepare_next_state(self, prediction, forcing=None):
        """
        Prepare the next state from model prediction.

        Combines prediction (prognostic vars) with forcing variables
        to create the input for the next forecast step.

        Args:
            prediction: Model output (scaled prognostic + diagnostic variables)
            forcing: Forcing variables for next timestep (scaled), optional

        Returns:
            Graph ready for next model forward pass
        """
        raise NotImplementedError(
            "State preparation logic needs to be implemented based on your "
            "specific graph structure and how you want to separate prognostic/"
            "diagnostic outputs and combine with forcing variables."
        )

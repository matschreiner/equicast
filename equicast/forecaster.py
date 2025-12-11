import torch


class Forecaster:
    """
    Simplified forecaster that delegates all preprocessing to the Model.

    The model handles scaling and feature routing internally, so the
    forecaster just manages the autoregressive loop.
    """

    def __init__(self, model: torch.nn.Module):
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
                    pred,
                    (
                        forcing_sequence[step]
                        if forcing_sequence is not None
                        else None
                    ),
                )

        return predictions

    def _prepare_next_state(self, prediction, forcing=None):
        """
        Prepare the next state from model prediction.

        Args:
            prediction: Model output
            forcing: Forcing variables for next timestep, optional

        Returns:
            Graph ready for next model forward pass
        """
        raise NotImplementedError(
            "State preparation logic needs to be implemented based on your "
            "specific graph structure and how you want to separate prognostic/"
            "diagnostic outputs and combine with forcing variables."
        )

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
                    current_state,
                    pred,
                    (
                        forcing_sequence[step]
                        if forcing_sequence is not None
                        else None
                    ),
                )

        return predictions

    def _prepare_next_state(self, current_graph, prediction, forcing=None):
        """
        Prepare the next state from model prediction.

        The prediction from the model contains [prognostic, diagnostic] variables
        in scaled space. We need to:
        1. Unscale the prediction
        2. Extract prognostic variables
        3. Combine with forcing data
        4. Create full feature vector for next timestep

        Args:
            current_graph: Current graph (used to clone structure)
            prediction: Model output [prognostic, diagnostic] in scaled space
            forcing: Forcing variables for next timestep in physical space, optional

        Returns:
            Graph ready for next model forward pass
        """
        data_handler = self.model.data_handler

        # Unscale prediction from normalized space to physical space
        pred_unscaled = data_handler.scaler.inverse_transform(prediction)

        # Split prediction: first N features are prognostic, rest are diagnostic
        n_prognostic = len(data_handler.feature_router.feature_config.prognostic)
        prognostic = pred_unscaled[:, :n_prognostic]

        # Create full feature vector for next timestep
        n_features = len(data_handler.name_to_index)
        next_state = torch.zeros(
            prognostic.shape[0], n_features, device=prediction.device
        )

        # Place prognostic variables at their correct indices
        prog_idxs = data_handler.feature_router._get_data_idxs(
            data_handler.feature_router.feature_config.prognostic
        )
        for i, idx in enumerate(prog_idxs):
            next_state[:, idx] = prognostic[:, i]

        # Place forcing variables at their correct indices
        if forcing is not None:
            forcing_idxs = data_handler.feature_router._get_data_idxs(
                data_handler.feature_router.feature_config.forcing
            )
            for i, idx in enumerate(forcing_idxs):
                next_state[:, idx] = forcing[:, i]

        # Clone graph structure and update input_state
        next_graph = current_graph.clone()
        next_graph["grid"].input_state = next_state

        return next_graph

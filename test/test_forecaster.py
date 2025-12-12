import pytest
import torch
from torch_geometric.data import HeteroData
from unittest.mock import Mock

from equicast.forecaster import Forecaster


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.eval = Mock()

    # Mock forward to return a simple prediction
    def forward_side_effect(graph):
        # Return prediction based on input
        return torch.randn(10, 3)

    model.return_value = torch.randn(10, 3)
    model.side_effect = forward_side_effect

    return model


@pytest.fixture
def sample_initial_state():
    """Create a sample initial state graph."""
    graph = HeteroData()
    graph["grid"].input_state = torch.randn(10, 5)
    return graph


def test_forecaster_initialization(mock_model):
    """Test Forecaster initialization."""
    forecaster = Forecaster(mock_model)

    assert forecaster.model is mock_model


def test_forecaster_forecast_calls_eval(mock_model, sample_initial_state):
    """Test that forecast puts model in eval mode."""
    forecaster = Forecaster(mock_model)

    # Mock _prepare_next_state to avoid NotImplementedError
    forecaster._prepare_next_state = Mock(return_value=sample_initial_state)

    forecaster.forecast(sample_initial_state, steps=1)

    mock_model.eval.assert_called_once()


def test_forecaster_forecast_single_step(mock_model, sample_initial_state):
    """Test forecasting for a single step."""
    forecaster = Forecaster(mock_model)

    # Mock _prepare_next_state to avoid NotImplementedError
    forecaster._prepare_next_state = Mock(return_value=sample_initial_state)

    predictions = forecaster.forecast(sample_initial_state, steps=1)

    assert len(predictions) == 1
    assert mock_model.call_count == 1


def test_forecaster_forecast_multiple_steps(mock_model, sample_initial_state):
    """Test forecasting for multiple steps."""
    forecaster = Forecaster(mock_model)

    # Mock _prepare_next_state
    forecaster._prepare_next_state = Mock(return_value=sample_initial_state)

    predictions = forecaster.forecast(sample_initial_state, steps=5)

    assert len(predictions) == 5
    assert mock_model.call_count == 5


def test_forecaster_forecast_autoregressive(mock_model, sample_initial_state):
    """Test that forecast is autoregressive (uses previous prediction)."""
    forecaster = Forecaster(mock_model)

    # Track states passed to _prepare_next_state
    state_history = []

    def prepare_next_state(graph, pred, forcing):
        state_history.append(pred)
        return sample_initial_state

    forecaster._prepare_next_state = Mock(side_effect=prepare_next_state)

    forecaster.forecast(sample_initial_state, steps=3)

    # Should call _prepare_next_state 3 times (once per step)
    assert forecaster._prepare_next_state.call_count == 3


def test_forecaster_forecast_with_no_forcing(mock_model, sample_initial_state):
    """Test forecast without forcing sequence."""
    forecaster = Forecaster(mock_model)

    calls = []
    def prepare_next_state(graph, pred, forcing):
        calls.append((pred, forcing))
        return sample_initial_state

    forecaster._prepare_next_state = Mock(side_effect=prepare_next_state)

    forecaster.forecast(sample_initial_state, steps=2, forcing_sequence=None)

    # All forcing should be None
    for pred, forcing in calls:
        assert forcing is None


def test_forecaster_forecast_with_forcing(mock_model, sample_initial_state):
    """Test forecast with forcing sequence."""
    forecaster = Forecaster(mock_model)

    forcing_seq = [torch.randn(10, 2) for _ in range(3)]
    calls = []

    def prepare_next_state(graph, pred, forcing):
        calls.append((pred, forcing))
        return sample_initial_state

    forecaster._prepare_next_state = Mock(side_effect=prepare_next_state)

    forecaster.forecast(sample_initial_state, steps=3, forcing_sequence=forcing_seq)

    # Should pass forcing for each step
    assert len(calls) == 3
    for i, (pred, forcing) in enumerate(calls):
        assert torch.equal(forcing, forcing_seq[i])


def test_forecaster_forecast_no_gradient(mock_model, sample_initial_state):
    """Test that forecast runs without gradients."""
    forecaster = Forecaster(mock_model)
    forecaster._prepare_next_state = Mock(return_value=sample_initial_state)

    # Enable gradient tracking
    sample_initial_state["grid"].input_state.requires_grad = True

    with torch.no_grad():
        predictions = forecaster.forecast(sample_initial_state, steps=2)

    # Predictions should not require gradients
    for pred in predictions:
        assert not pred.requires_grad


def test_forecaster_prepare_next_state_implementation():
    """Test that _prepare_next_state delegates to DataHandler correctly."""
    from unittest.mock import Mock

    # Create mock model with data_handler
    mock_model = Mock()
    mock_data_handler = Mock()

    # Mock extract_prognostic to return prognostic variables
    mock_prognostic = torch.tensor([[1.0, 2.0]] * 5)  # 5 nodes, 2 prognostic vars
    mock_data_handler.extract_prognostic = Mock(return_value=mock_prognostic)

    # Mock reconstruct_state to return full state
    mock_full_state = torch.tensor([[0.5, 1.0, 2.0, 0.0]] * 5)  # 5 nodes, 4 features
    mock_data_handler.reconstruct_state = Mock(return_value=mock_full_state)

    mock_model.data_handler = mock_data_handler

    forecaster = Forecaster(mock_model)

    # Create sample graph and prediction
    sample_graph = HeteroData()
    sample_graph["grid"].input_state = torch.ones(5, 4)

    # Prediction has [prognostic, diagnostic] = [temp, humidity, precip]
    prediction = torch.tensor([[1.0, 2.0, 3.0]] * 5)  # 5 nodes, 3 outputs
    forcing = torch.tensor([[0.5]] * 5)  # 5 nodes, 1 forcing variable

    next_graph = forecaster._prepare_next_state(sample_graph, prediction, forcing)

    # Check that reconstruct_state was called with named arguments
    mock_data_handler.reconstruct_state.assert_called_once()
    call_kwargs = mock_data_handler.reconstruct_state.call_args.kwargs
    assert torch.equal(call_kwargs["prediction"], prediction)
    assert torch.equal(call_kwargs["forcing"], forcing)

    # Check that next_state was set correctly
    assert torch.equal(next_graph["grid"].input_state, mock_full_state)


def test_forecaster_returns_stacked_tensor(mock_model, sample_initial_state):
    """Test that forecast returns a stacked tensor."""
    forecaster = Forecaster(mock_model)
    forecaster._prepare_next_state = Mock(return_value=sample_initial_state)

    predictions = forecaster.forecast(sample_initial_state, steps=3)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape[0] == 3  # First dimension is time

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

    try:
        forecaster.forecast(sample_initial_state, steps=1)
    except NotImplementedError:
        pass  # Expected for _prepare_next_state

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

    def prepare_next_state(pred, forcing):
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
    def prepare_next_state(pred, forcing):
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

    def prepare_next_state(pred, forcing):
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


def test_forecaster_prepare_next_state_not_implemented():
    """Test that _prepare_next_state raises NotImplementedError by default."""
    forecaster = Forecaster(Mock())

    with pytest.raises(NotImplementedError) as exc_info:
        forecaster._prepare_next_state(torch.randn(10, 3), None)

    assert "State preparation logic needs to be implemented" in str(exc_info.value)


def test_forecaster_returns_list(mock_model, sample_initial_state):
    """Test that forecast returns a list of predictions."""
    forecaster = Forecaster(mock_model)
    forecaster._prepare_next_state = Mock(return_value=sample_initial_state)

    predictions = forecaster.forecast(sample_initial_state, steps=3)

    assert isinstance(predictions, list)
    assert len(predictions) == 3

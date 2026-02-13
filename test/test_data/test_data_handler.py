import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from equicast.data.data_handler import DataHandler
from equicast.data.feature_config import FeatureConfig


@pytest.fixture
def sample_feature_config():
    """Create sample feature configuration."""
    return FeatureConfig(
        forcing=["solar", "pressure"],
        prognostic=["temp", "humidity"],
        diagnostic=["precip"],
    )


@pytest.fixture
def mock_dataset():
    """Mock anemoi dataset."""
    dataset = Mock()
    # Return numpy arrays which cast_dict can convert to tensors
    import numpy as np

    type(dataset).statistics = {
        "mean": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "stdev": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    }
    type(dataset).name_to_index = {
        "solar": 0,
        "pressure": 1,
        "temp": 2,
        "humidity": 3,
        "precip": 4,
    }
    return dataset


def test_data_handler_initialization(sample_feature_config, mock_dataset):
    """Test DataHandler initialization."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        assert handler.dataset_path == "fake_path.zarr"
        assert handler.feature_config == sample_feature_config
        assert handler.scaler is not None
        assert handler.feature_router is not None


def test_data_handler_statistics(sample_feature_config, mock_dataset):
    """Test that statistics are properly extracted and converted."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        assert "mean" in handler.statistics
        assert "stdev" in handler.statistics
        assert isinstance(handler.statistics["mean"], torch.Tensor)
        assert isinstance(handler.statistics["stdev"], torch.Tensor)


def test_data_handler_name_to_index(sample_feature_config, mock_dataset):
    """Test that name_to_index mapping is stored."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        assert handler.name_to_index == mock_dataset.name_to_index
        assert isinstance(handler.name_to_index, dict)


def test_data_handler_in_idxs_property(sample_feature_config, mock_dataset):
    """Test in_idxs property delegates to feature_router."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # forcing: [0, 1], prognostic: [2, 3]
        expected = [0, 1, 2, 3]
        assert handler.in_idxs == expected
        assert handler.in_idxs == handler.feature_router.in_idxs


def test_data_handler_out_idxs_property(sample_feature_config, mock_dataset):
    """Test out_idxs property delegates to feature_router."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # prognostic: [2, 3], diagnostic: [4]
        expected = [2, 3, 4]
        assert handler.out_idxs == expected
        assert handler.out_idxs == handler.feature_router.out_idxs


def test_data_handler_scaler_works(sample_feature_config, mock_dataset):
    """Test that the scaler can transform data."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        data = torch.randn(10, 5)
        scaled = handler.scaler.transform(data)
        recovered = handler.scaler.inverse_transform(scaled)

        assert torch.allclose(recovered, data, atol=1e-6)


def test_data_handler_feature_router_works(sample_feature_config, mock_dataset):
    """Test that the feature_router has correct indices."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        assert len(handler.in_idxs) == 4  # 2 forcing + 2 prognostic
        assert len(handler.out_idxs) == 3  # 2 prognostic + 1 diagnostic


def test_data_handler_extract_prognostic(sample_feature_config, mock_dataset):
    """Test extracting prognostic variables from prediction."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # Prediction: [temp, humidity, precip] = [prognostic, prognostic, diagnostic]
        # out_idxs = [2, 3, 4] (temp at idx 2, humidity at idx 3, precip at idx 4)
        # mean = [0,1,2,3,4], std = [1,1,1,1,1]
        # mean_out = [2, 3, 4], std_out = [1, 1, 1]
        # Prediction in scaled space: [10, 19, 28]
        prediction_scaled = torch.tensor([[10.0, 19.0, 28.0]] * 5)  # 5 nodes

        prognostic = handler.extract_prognostic(prediction_scaled)

        # Should extract only first 2 features (prognostic) and unscale them
        assert prognostic.shape == (5, 2)
        # After unscaling: pred * std_out + mean_out
        # [10*1+2, 19*1+3, 28*1+4] = [12, 22, 32]
        # Extract prognostic (first 2): [12, 22]
        assert torch.allclose(prognostic[:, 0], torch.tensor(12.0))
        assert torch.allclose(prognostic[:, 1], torch.tensor(22.0))


def test_data_handler_reconstruct_state(sample_feature_config, mock_dataset):
    """Test reconstructing full state from prognostic and forcing."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # Prognostic: [temp, humidity] at indices [2, 3]
        prognostic = torch.tensor([[10.0, 20.0]] * 5)

        # Forcing: [solar, pressure] at indices [0, 1]
        forcing = torch.tensor([[1.0, 2.0]] * 5)

        state = handler.reconstruct_state(prognostic=prognostic, forcing=forcing)

        # Should have shape [5 nodes, 5 features]
        assert state.shape == (5, 5)

        # Check values are placed correctly
        assert torch.allclose(state[:, 0], torch.tensor(1.0))  # solar
        assert torch.allclose(state[:, 1], torch.tensor(2.0))  # pressure
        assert torch.allclose(state[:, 2], torch.tensor(10.0))  # temp
        assert torch.allclose(state[:, 3], torch.tensor(20.0))  # humidity
        assert torch.allclose(state[:, 4], torch.tensor(0.0))  # precip (not provided)


def test_data_handler_reconstruct_state_no_forcing(sample_feature_config, mock_dataset):
    """Test reconstructing state without forcing variables."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # Only prognostic
        prognostic = torch.tensor([[10.0, 20.0]] * 5)

        state = handler.reconstruct_state(prognostic=prognostic)

        # Prognostic should be placed correctly, forcing should be zeros
        assert state.shape == (5, 5)
        assert torch.allclose(state[:, 0], torch.tensor(0.0))  # solar (not provided)
        assert torch.allclose(state[:, 1], torch.tensor(0.0))  # pressure (not provided)
        assert torch.allclose(state[:, 2], torch.tensor(10.0))  # temp
        assert torch.allclose(state[:, 3], torch.tensor(20.0))  # humidity


def test_data_handler_prepare_model_input(sample_feature_config, mock_dataset):
    """Test preparing raw state for model input."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # Raw state: [solar, pressure, temp, humidity, precip]
        # mean = [0, 1, 2, 3, 4], std = [1, 1, 1, 1, 1]
        raw_state = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]] * 10)

        # Prepare model input (scale + route to forcing + prognostic)
        model_input = handler.prepare_model_input(raw_state)

        # Should have shape [10 nodes, 4 input features]
        # in_idxs = [0, 1, 2, 3] (forcing + prognostic)
        assert model_input.shape == (10, 4)

        # Values should be scaled: (raw - mean) / std
        # Index 0 (solar): (1.0 - 0.0) / 1.0 = 1.0
        # Index 1 (pressure): (2.0 - 1.0) / 1.0 = 1.0
        # Index 2 (temp): (3.0 - 2.0) / 1.0 = 1.0
        # Index 3 (humidity): (4.0 - 3.0) / 1.0 = 1.0
        expected = torch.tensor([[1.0, 1.0, 1.0, 1.0]] * 10)
        assert torch.allclose(model_input, expected)


def test_data_handler_prepare_model_target(sample_feature_config, mock_dataset):
    """Test preparing raw state for model target."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # Raw state: [solar, pressure, temp, humidity, precip]
        raw_state = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]] * 10)

        # Prepare model target (scale + route to prognostic + diagnostic)
        model_target = handler.prepare_model_target(raw_state)

        # Should have shape [10 nodes, 3 output features]
        # out_idxs = [2, 3, 4] (prognostic + diagnostic)
        assert model_target.shape == (10, 3)

        # Values should be scaled
        # Index 2 (temp): (3.0 - 2.0) / 1.0 = 1.0
        # Index 3 (humidity): (4.0 - 3.0) / 1.0 = 1.0
        # Index 4 (precip): (5.0 - 4.0) / 1.0 = 1.0
        expected = torch.tensor([[1.0, 1.0, 1.0]] * 10)
        assert torch.allclose(model_target, expected)


def test_data_handler_from_prediction(sample_feature_config, mock_dataset):
    """Test converting model output back to raw state using prediction argument."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # Model output: [prognostic, diagnostic] in scaled space
        # out_idxs = [2, 3, 4] (temp, humidity, precip)
        # This is scaled, so we need to set it up correctly
        model_output = torch.tensor([[10.0, 19.0, 28.0]] * 5)

        # Forcing in physical space
        forcing = torch.tensor([[1.0, 2.0]] * 5)

        # Convert back to raw state using prediction argument
        raw_state = handler.reconstruct_state(prediction=model_output, forcing=forcing)

        # Should have shape [5 nodes, 5 features]
        assert raw_state.shape == (5, 5)

        # Forcing should be placed correctly (physical space, unchanged)
        assert torch.allclose(raw_state[:, 0], torch.tensor(1.0))  # solar
        assert torch.allclose(raw_state[:, 1], torch.tensor(2.0))  # pressure

        # Prognostic should be unscaled:
        # pred * std_out + mean_out
        # [10*1+2, 19*1+3] = [12, 22]
        assert torch.allclose(raw_state[:, 2], torch.tensor(12.0))  # temp
        assert torch.allclose(raw_state[:, 3], torch.tensor(22.0))  # humidity

        # Diagnostic not in prognostic, should be zero
        assert torch.allclose(raw_state[:, 4], torch.tensor(0.0))  # precip


def test_data_handler_from_prediction_no_forcing(sample_feature_config, mock_dataset):
    """Test converting model output without forcing."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        model_output = torch.tensor([[10.0, 19.0, 28.0]] * 5)

        raw_state = handler.reconstruct_state(prediction=model_output)

        # Should have zeros for forcing
        assert raw_state.shape == (5, 5)
        assert torch.allclose(raw_state[:, 0], torch.tensor(0.0))  # solar (not provided)
        assert torch.allclose(raw_state[:, 1], torch.tensor(0.0))  # pressure (not provided)

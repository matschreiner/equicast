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
    with patch('equicast.data.data_handler.open_dataset', return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        assert handler.dataset_path == "fake_path.zarr"
        assert handler.feature_config == sample_feature_config
        assert handler.scaler is not None
        assert handler.feature_router is not None


def test_data_handler_statistics(sample_feature_config, mock_dataset):
    """Test that statistics are properly extracted and converted."""
    with patch('equicast.data.data_handler.open_dataset', return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        assert "mean" in handler.statistics
        assert "stdev" in handler.statistics
        assert isinstance(handler.statistics["mean"], torch.Tensor)
        assert isinstance(handler.statistics["stdev"], torch.Tensor)


def test_data_handler_name_to_index(sample_feature_config, mock_dataset):
    """Test that name_to_index mapping is stored."""
    with patch('equicast.data.data_handler.open_dataset', return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        assert handler.name_to_index == mock_dataset.name_to_index
        assert isinstance(handler.name_to_index, dict)


def test_data_handler_in_idxs_property(sample_feature_config, mock_dataset):
    """Test in_idxs property delegates to feature_router."""
    with patch('equicast.data.data_handler.open_dataset', return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # forcing: [0, 1], prognostic: [2, 3]
        expected = [0, 1, 2, 3]
        assert handler.in_idxs == expected
        assert handler.in_idxs == handler.feature_router.in_idxs


def test_data_handler_out_idxs_property(sample_feature_config, mock_dataset):
    """Test out_idxs property delegates to feature_router."""
    with patch('equicast.data.data_handler.open_dataset', return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        # prognostic: [2, 3], diagnostic: [4]
        expected = [2, 3, 4]
        assert handler.out_idxs == expected
        assert handler.out_idxs == handler.feature_router.out_idxs


def test_data_handler_scaler_works(sample_feature_config, mock_dataset):
    """Test that the scaler can transform data."""
    with patch('equicast.data.data_handler.open_dataset', return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        data = torch.randn(10, 5)
        scaled = handler.scaler.transform(data)
        recovered = handler.scaler.inverse_transform(scaled)

        assert torch.allclose(recovered, data, atol=1e-6)


def test_data_handler_feature_router_works(sample_feature_config, mock_dataset):
    """Test that the feature_router has correct indices."""
    with patch('equicast.data.data_handler.open_dataset', return_value=mock_dataset):
        handler = DataHandler("fake_path.zarr", sample_feature_config)

        assert len(handler.in_idxs) == 4  # 2 forcing + 2 prognostic
        assert len(handler.out_idxs) == 3  # 2 prognostic + 1 diagnostic

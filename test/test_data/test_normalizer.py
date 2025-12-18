import pytest
import torch

from equicast.data.scaler import Scaler


@pytest.fixture
def sample_statistics():
    """Create sample statistics for testing."""
    return {
        "mean": torch.tensor([0.0, 1.0, 2.0]),
        "stdev": torch.tensor([1.0, 2.0, 0.5]),
    }


@pytest.fixture
def scaler(sample_statistics):
    """Create a Scaler instance."""
    return Scaler(sample_statistics)


def test_scaler_initialization(sample_statistics):
    """Test Scaler initialization."""
    scaler = Scaler(sample_statistics)

    assert torch.equal(scaler.mean, sample_statistics["mean"])
    assert torch.equal(scaler.std, sample_statistics["stdev"])
    assert scaler.statistics == sample_statistics


def test_scaler_transform(scaler):
    """Test scaling transformation (z-score normalization)."""
    data = torch.tensor([0.0, 1.0, 2.0])

    scaled = scaler.transform(data)

    # Expected: (data - mean) / std
    expected = torch.tensor([0.0, 0.0, 0.0])
    assert torch.allclose(scaled, expected)


def test_scaler_transform_different_values(scaler):
    """Test scaling with different values."""
    data = torch.tensor([1.0, 5.0, 3.0])

    scaled = scaler.transform(data)

    # Expected: (data - mean) / std
    # (1.0 - 0.0) / 1.0 = 1.0
    # (5.0 - 1.0) / 2.0 = 2.0
    # (3.0 - 2.0) / 0.5 = 2.0
    expected = torch.tensor([1.0, 2.0, 2.0])
    assert torch.allclose(scaled, expected)


def test_scaler_inverse_transform(scaler):
    """Test inverse transformation."""
    scaled_data = torch.tensor([1.0, 2.0, 2.0])

    data = scaler.inverse_transform(scaled_data)

    # Expected: scaled * std + mean
    expected = torch.tensor([1.0, 5.0, 3.0])
    assert torch.allclose(data, expected)


def test_scaler_roundtrip(scaler):
    """Test that transform -> inverse_transform returns original data."""
    original = torch.tensor([1.5, 3.7, 2.3])

    scaled = scaler.transform(original)
    recovered = scaler.inverse_transform(scaled)

    assert torch.allclose(recovered, original, atol=1e-6)


def test_scaler_call_method(scaler):
    """Test that calling scaler directly uses transform."""
    data = torch.tensor([1.0, 5.0, 3.0])

    result = scaler(data)
    expected = scaler.transform(data)

    assert torch.equal(result, expected)


def test_scaler_multidimensional(scaler):
    """Test scaling with multidimensional tensors."""
    data = torch.tensor([
        [0.0, 1.0, 2.0],
        [1.0, 5.0, 3.0],
    ])

    scaled = scaler.transform(data)
    recovered = scaler.inverse_transform(scaled)

    assert torch.allclose(recovered, data, atol=1e-6)


def test_scaler_batch(scaler):
    """Test scaling with batch dimension."""
    batch_size = 4
    data = torch.randn(batch_size, 3)

    scaled = scaler.transform(data)
    recovered = scaler.inverse_transform(scaled)

    assert scaled.shape == data.shape
    assert torch.allclose(recovered, data, atol=1e-6)

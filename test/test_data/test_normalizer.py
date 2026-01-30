import numpy as np
import pytest
import torch

from equicast.data.normalizer import Normalizer


@pytest.fixture
def sample_statistics():
    """Create sample statistics for testing."""
    return {
        "mean": torch.tensor([0.0, 1.0, 2.0]),
        "stdev": torch.tensor([1.0, 2.0, 0.5]),
    }


@pytest.fixture
def normalizer(sample_statistics):
    """Create a Normalizer instance."""
    return Normalizer(sample_statistics)


def test_normalizer_initialization(sample_statistics):
    """Test Normalizer initialization."""
    normalizer = Normalizer(sample_statistics)

    assert torch.equal(normalizer.mean, sample_statistics["mean"])
    assert torch.equal(normalizer.std, sample_statistics["stdev"])
    assert normalizer.statistics == sample_statistics


def test_normalizer_transform(normalizer):
    """Test scaling transformation (z-score normalization)."""
    data = torch.tensor([0.0, 1.0, 2.0])

    scaled = normalizer.transform(data)

    # Expected: (data - mean) / std
    expected = torch.tensor([0.0, 0.0, 0.0])
    assert torch.allclose(scaled, expected)


def test_normalizer_transform_different_values(normalizer):
    """Test scaling with different values."""
    data = torch.tensor([1.0, 5.0, 3.0])

    scaled = normalizer.transform(data)

    # Expected: (data - mean) / std
    # (1.0 - 0.0) / 1.0 = 1.0
    # (5.0 - 1.0) / 2.0 = 2.0
    # (3.0 - 2.0) / 0.5 = 2.0
    expected = torch.tensor([1.0, 2.0, 2.0])
    assert torch.allclose(scaled, expected)


def test_normalizer_inverse_transform(normalizer):
    """Test inverse transformation."""
    scaled_data = torch.tensor([1.0, 2.0, 2.0])

    data = normalizer.inverse_transform(scaled_data)

    # Expected: scaled * std + mean
    expected = torch.tensor([1.0, 5.0, 3.0])
    assert torch.allclose(data, expected)


def test_normalizer_roundtrip(normalizer):
    """Test that transform -> inverse_transform returns original data."""
    original = torch.tensor([1.5, 3.7, 2.3])

    scaled = normalizer.transform(original)
    recovered = normalizer.inverse_transform(scaled)

    assert torch.allclose(recovered, original, atol=1e-6)


def test_normalizer_multidimensional(normalizer):
    """Test scaling with multidimensional tensors."""
    data = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [1.0, 5.0, 3.0],
        ]
    )

    scaled = normalizer.transform(data)
    recovered = normalizer.inverse_transform(scaled)

    assert torch.allclose(recovered, data, atol=1e-6)


def test_normalizer_batch(normalizer):
    """Test scaling with batch dimension."""
    batch_size = 4
    data = torch.randn(batch_size, 3)

    scaled = normalizer.transform(data)
    recovered = normalizer.inverse_transform(scaled)

    assert scaled.shape == data.shape
    assert torch.allclose(recovered, data, atol=1e-6)


class MockDataset:
    """Mock dataset for testing vector statistics computation."""

    def __init__(self, data: np.ndarray):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def test_compute_vector_statistics(normalizer):
    """Test that compute_vector_statistics computes correct mean norms."""
    # Create dataset with known vector values
    # 2 samples, 4 nodes, 4 features (u1, v1, u2, v2)
    data = np.array(
        [
            # Sample 0: vectors with norm 5 (3,4) and norm 13 (5,12)
            [[3, 4, 5, 12]] * 4,  # 4 nodes, same values
            # Sample 1: vectors with norm 5 (4,3) and norm 13 (12,5)
            [[4, 3, 12, 5]] * 4,
        ],
        dtype=np.float32,
    )

    dataset = MockDataset(data)
    vector_indices = [(0, 1), (2, 3)]  # (u1, v1), (u2, v2)

    normalizer.compute_vector_statistics(dataset, vector_indices, num_samples=2)

    # Both vectors should have consistent norms: 5 and 13
    assert torch.allclose(normalizer.vector_mean_norm[0], torch.tensor(5.0))
    assert torch.allclose(normalizer.vector_mean_norm[1], torch.tensor(13.0))


def test_transform_vector_roundtrip(normalizer):
    """Test that transform_vector and inverse_transform_vector are inverses."""
    # Setup vector statistics
    data = np.array(
        [
            [[3, 4, 5, 12]] * 4,
            [[4, 3, 12, 5]] * 4,
        ],
        dtype=np.float32,
    )
    dataset = MockDataset(data)
    normalizer.compute_vector_statistics(
        dataset, [(0, 1), (2, 3)], num_samples=2
    )

    # Test roundtrip
    original = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    transformed = normalizer.transform_vector(original, vector_idx=0)
    recovered = normalizer.inverse_transform_vector(transformed, vector_idx=0)

    assert torch.allclose(recovered, original)


def test_transform_vector_normalizes_to_unit_average(normalizer):
    """Test that transformed vectors have average norm close to 1."""
    # Setup with known mean norm of 5
    data = np.array(
        [
            [[3, 4, 0, 0]] * 4,
            [[4, 3, 0, 0]] * 4,
        ],
        dtype=np.float32,
    )
    dataset = MockDataset(data)
    normalizer.compute_vector_statistics(dataset, [(0, 1)], num_samples=2)

    # Transform a vector with norm 5
    original = torch.tensor([3.0, 4.0])
    transformed = normalizer.transform_vector(original, vector_idx=0)

    # Transformed norm should be 1 (since original norm equals mean norm)
    transformed_norm = torch.sqrt((transformed**2).sum())
    assert torch.allclose(transformed_norm, torch.tensor(1.0))

import pytest
import torch

from equicast.data.feature_config import FeatureConfig
from equicast.data.feature_indices import FeatureIndices


@pytest.fixture
def sample_feature_config():
    """Create sample feature configuration."""
    return FeatureConfig(
        forcing=["solar", "pressure"],
        prognostic=["temp", "humidity"],
        diagnostic=["precip"],
    )


@pytest.fixture
def sample_name_to_index():
    """Create sample name to index mapping."""
    return {
        "solar": 0,
        "pressure": 1,
        "temp": 2,
        "humidity": 3,
        "precip": 4,
    }


@pytest.fixture
def feature_indices(sample_feature_config, sample_name_to_index):
    """Create FeatureIndices instance."""
    return FeatureIndices(sample_feature_config, sample_name_to_index)


def test_feature_indices_initialization(
    feature_indices, sample_feature_config, sample_name_to_index
):
    """Test FeatureIndices initialization."""
    assert feature_indices.feature_config == sample_feature_config
    assert feature_indices.name_to_index == sample_name_to_index


def test_feature_indices_in_idxs(feature_indices):
    """Test input indices calculation (forcing + prognostic)."""
    # forcing: [0, 1], prognostic: [2, 3]
    expected = [0, 1, 2, 3]
    assert feature_indices.in_idxs == expected


def test_feature_indices_out_idxs(feature_indices):
    """Test output indices calculation (prognostic + diagnostic)."""
    # prognostic: [2, 3], diagnostic: [4]
    expected = [2, 3, 4]
    assert feature_indices.out_idxs == expected


def test_feature_indices_get_data_idxs(feature_indices):
    """Test _get_data_idxs helper method."""
    names = ["temp", "solar", "precip"]
    indices = feature_indices._get_data_idxs(names)

    assert indices == [2, 0, 4]


def test_feature_indices_with_different_order():
    """Test FeatureIndices with features in different order."""
    config = FeatureConfig(
        forcing=["b"],
        prognostic=["a", "c"],
        diagnostic=["d"],
    )
    name_to_index = {"a": 0, "b": 1, "c": 2, "d": 3}

    indices = FeatureIndices(config, name_to_index)

    # forcing: [1], prognostic: [0, 2]
    assert indices.in_idxs == [1, 0, 2]
    # prognostic: [0, 2], diagnostic: [3]
    assert indices.out_idxs == [0, 2, 3]


def test_feature_indices_indices_are_lists(feature_indices):
    """Test that indices are stored as lists."""
    assert isinstance(feature_indices.in_idxs, list)
    assert isinstance(feature_indices.out_idxs, list)


def test_feature_indices_empty_forcing():
    """Test FeatureIndices with no forcing variables."""
    config = FeatureConfig(
        forcing=[],
        prognostic=["a", "b"],
        diagnostic=["c"],
    )
    name_to_index = {"a": 0, "b": 1, "c": 2}

    indices = FeatureIndices(config, name_to_index)

    # Only prognostic in input
    assert indices.in_idxs == [0, 1]
    assert indices.out_idxs == [0, 1, 2]


def test_feature_indices_empty_diagnostic():
    """Test FeatureIndices with no diagnostic variables."""
    config = FeatureConfig(
        forcing=["a"],
        prognostic=["b"],
        diagnostic=[],
    )
    name_to_index = {"a": 0, "b": 1}

    indices = FeatureIndices(config, name_to_index)

    assert indices.in_idxs == [0, 1]
    # Only prognostic in output
    assert indices.out_idxs == [1]

import pytest
import torch

from equicast.data.feature_config import FeatureConfig
from equicast.data.feature_router import FeatureRouter


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
def feature_router(sample_feature_config, sample_name_to_index):
    """Create FeatureRouter instance."""
    return FeatureRouter(sample_feature_config, sample_name_to_index)


def test_feature_router_initialization(feature_router, sample_feature_config, sample_name_to_index):
    """Test FeatureRouter initialization."""
    assert feature_router.feature_config == sample_feature_config
    assert feature_router.name_to_index == sample_name_to_index


def test_feature_router_in_idxs(feature_router):
    """Test input indices calculation (forcing + prognostic)."""
    # forcing: [0, 1], prognostic: [2, 3]
    expected = [0, 1, 2, 3]
    assert feature_router.in_idxs == expected


def test_feature_router_out_idxs(feature_router):
    """Test output indices calculation (prognostic + diagnostic)."""
    # prognostic: [2, 3], diagnostic: [4]
    expected = [2, 3, 4]
    assert feature_router.out_idxs == expected


def test_feature_router_get_data_idxs(feature_router):
    """Test _get_data_idxs helper method."""
    names = ["temp", "solar", "precip"]
    indices = feature_router._get_data_idxs(names)

    assert indices == [2, 0, 4]


def test_feature_router_with_different_order():
    """Test FeatureRouter with features in different order."""
    config = FeatureConfig(
        forcing=["b"],
        prognostic=["a", "c"],
        diagnostic=["d"],
    )
    name_to_index = {"a": 0, "b": 1, "c": 2, "d": 3}

    router = FeatureRouter(config, name_to_index)

    # forcing: [1], prognostic: [0, 2]
    assert router.in_idxs == [1, 0, 2]
    # prognostic: [0, 2], diagnostic: [3]
    assert router.out_idxs == [0, 2, 3]


def test_feature_router_indices_are_lists(feature_router):
    """Test that indices are stored as lists."""
    assert isinstance(feature_router.in_idxs, list)
    assert isinstance(feature_router.out_idxs, list)


def test_feature_router_empty_forcing():
    """Test FeatureRouter with no forcing variables."""
    config = FeatureConfig(
        forcing=[],
        prognostic=["a", "b"],
        diagnostic=["c"],
    )
    name_to_index = {"a": 0, "b": 1, "c": 2}

    router = FeatureRouter(config, name_to_index)

    # Only prognostic in input
    assert router.in_idxs == [0, 1]
    assert router.out_idxs == [0, 1, 2]


def test_feature_router_empty_diagnostic():
    """Test FeatureRouter with no diagnostic variables."""
    config = FeatureConfig(
        forcing=["a"],
        prognostic=["b"],
        diagnostic=[],
    )
    name_to_index = {"a": 0, "b": 1}

    router = FeatureRouter(config, name_to_index)

    assert router.in_idxs == [0, 1]
    # Only prognostic in output
    assert router.out_idxs == [1]

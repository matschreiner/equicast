import pytest
import tempfile
import yaml
from pathlib import Path

from equicast.data.feature_config import FeatureConfig


@pytest.fixture
def sample_feature_dict():
    """Sample feature configuration dictionary."""
    return {
        "forcing": ["solar_radiation", "surface_pressure"],
        "prognostic": ["temperature", "humidity", "wind_u", "wind_v"],
        "diagnostic": ["precipitation", "cloud_cover"],
    }


@pytest.fixture
def sample_yaml_file(sample_feature_dict):
    """Create a temporary YAML file with feature config."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_feature_dict, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_feature_config_initialization(sample_feature_dict):
    """Test FeatureConfig initialization."""
    config = FeatureConfig(**sample_feature_dict)

    assert config.forcing == sample_feature_dict["forcing"]
    assert config.prognostic == sample_feature_dict["prognostic"]
    assert config.diagnostic == sample_feature_dict["diagnostic"]


def test_feature_config_from_yaml(sample_yaml_file, sample_feature_dict):
    """Test loading FeatureConfig from YAML file."""
    config = FeatureConfig.from_yaml(sample_yaml_file)

    assert config.forcing == sample_feature_dict["forcing"]
    assert config.prognostic == sample_feature_dict["prognostic"]
    assert config.diagnostic == sample_feature_dict["diagnostic"]


def test_feature_config_repr(sample_feature_dict):
    """Test FeatureConfig string representation."""
    config = FeatureConfig(**sample_feature_dict)

    repr_str = repr(config)

    assert "FeatureConfig" in repr_str
    assert "forcing=2" in repr_str
    assert "prognostic=4" in repr_str
    assert "diagnostic=2" in repr_str


def test_feature_config_attributes_are_lists():
    """Test that feature lists are stored as lists."""
    config = FeatureConfig(
        forcing=["a", "b"],
        prognostic=["c", "d", "e"],
        diagnostic=["f"],
    )

    assert isinstance(config.forcing, list)
    assert isinstance(config.prognostic, list)
    assert isinstance(config.diagnostic, list)


def test_feature_config_empty_lists():
    """Test FeatureConfig with empty lists."""
    config = FeatureConfig(
        forcing=[],
        prognostic=["temp"],
        diagnostic=[],
    )

    assert config.forcing == []
    assert len(config.prognostic) == 1
    assert config.diagnostic == []

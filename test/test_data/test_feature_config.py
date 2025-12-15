import tempfile
from pathlib import Path

import pytest
import yaml

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
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
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


@pytest.fixture
def sample_feature_dict_with_vectors():
    """Sample feature configuration with vector features."""
    return {
        "forcing": ["cos_lat", "sin_lat"],
        "prognostic": ["t2m", "sp", "z500"],
        "diagnostic": ["tp"],
        "prognostic_vector": {
            "wind_10m": ["u10", "v10"],
            "wind_850": ["u850", "v850"],
        },
    }


@pytest.fixture
def sample_yaml_file_with_vectors(sample_feature_dict_with_vectors):
    """Create a temporary YAML file with vector features."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        yaml.dump(sample_feature_dict_with_vectors, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_feature_config_with_vectors_from_yaml(sample_yaml_file_with_vectors):
    """Test loading FeatureConfig with vector features from YAML."""
    config = FeatureConfig.from_yaml(sample_yaml_file_with_vectors)

    assert config.forcing == ["cos_lat", "sin_lat"]
    assert config.prognostic == ["t2m", "sp", "z500"]
    assert config.diagnostic == ["tp"]
    assert "wind_10m" in config.prognostic_vector
    assert config.prognostic_vector["wind_10m"] == ["u10", "v10"]
    assert "wind_850" in config.prognostic_vector
    assert config.prognostic_vector["wind_850"] == ["u850", "v850"]


def test_feature_config_vectors_default_to_empty():
    """Test that vector fields default to empty dicts."""
    config = FeatureConfig(
        forcing=["a"],
        prognostic=["b"],
        diagnostic=["c"],
    )

    assert config.forcing_vector == {}
    assert config.prognostic_vector == {}
    assert config.diagnostic_vector == {}


def test_feature_config_with_all_vector_types():
    """Test FeatureConfig with forcing, prognostic, and diagnostic vectors."""
    config = FeatureConfig(
        forcing=["scalar_forcing"],
        prognostic=["scalar_prog"],
        diagnostic=["scalar_diag"],
        forcing_vector={"pos": ["x", "y"]},
        prognostic_vector={"wind": ["u", "v"]},
        diagnostic_vector={"flux": ["fx", "fy"]},
    )

    assert config.forcing_vector == {"pos": ["x", "y"]}
    assert config.prognostic_vector == {"wind": ["u", "v"]}
    assert config.diagnostic_vector == {"flux": ["fx", "fy"]}


def test_feature_config_repr_with_vectors():
    """Test FeatureConfig repr includes vector count."""
    config = FeatureConfig(
        forcing=["a"],
        prognostic=["b", "c"],
        diagnostic=["d"],
        prognostic_vector={"wind_1": ["u1", "v1"], "wind_2": ["u2", "v2"]},
    )

    repr_str = repr(config)

    assert "FeatureConfig" in repr_str
    assert "forcing=1" in repr_str
    assert "prognostic=2" in repr_str
    assert "diagnostic=1" in repr_str
    assert "vectors=2" in repr_str

import pytest
import torch
from torch_geometric.data import HeteroData

from equicast.data.equivariant_data_handler import MultiFrameEquivariantGraphDataHandler
from equicast.data.feature_config import FeatureConfig

DATASET_PATH = "test/res/micro_aifs.zarr"
GRAPH_PATH = "test/res/micro_aifs.pt"


@pytest.fixture
def feature_config():
    return FeatureConfig(
        forcing=["lsm", "cos_julian_day"],
        prognostic=["2t", "msl"],
        diagnostic=["cp"],
        prognostic_vector={"wind_10m": ["10u", "10v"]},
    )


@pytest.fixture
def handler(feature_config):
    return MultiFrameEquivariantGraphDataHandler(
        dataset_path=DATASET_PATH,
        feature_config=feature_config,
        n_input_frames=2,
    )


@pytest.fixture
def single_handler(feature_config):
    from equicast.data.equivariant_data_handler import EquivariantGraphDataHandler
    return EquivariantGraphDataHandler(
        dataset_path=DATASET_PATH,
        feature_config=feature_config,
    )


@pytest.fixture
def batch(graph_provider):
    from equicast.data import AnemoiDataset
    dataset = AnemoiDataset(path=DATASET_PATH, graph_provider=graph_provider, no_frames=3)
    return dataset[0]


def test_in_dim_scales_with_n_frames(handler, single_handler):
    assert handler.in_dim == 2 * single_handler.in_dim


def test_in_vector_dim_scales_with_n_frames(handler, single_handler):
    assert handler.in_vector_dim == 2 * single_handler.in_vector_dim


def test_prepare_training_batch_returns_tuple(handler, batch):
    result = handler.prepare_training_batch(batch)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_input_scalar_shape(handler, single_handler, batch):
    backbone_input, _ = handler.prepare_training_batch(batch)
    nodes = backbone_input[handler.nodes]
    assert nodes["input_scalar"].shape[-1] == 2 * single_handler.in_dim


def test_input_vector_shape(handler, single_handler, batch):
    backbone_input, _ = handler.prepare_training_batch(batch)
    nodes = backbone_input[handler.nodes]
    assert nodes["input_vector"].shape[-2] == 2 * single_handler.in_vector_dim


def test_target_has_scalar_and_vector(handler, batch):
    _, target = handler.prepare_training_batch(batch)
    assert "scalar" in target
    assert "vector" in target


def test_residual_comes_from_last_input_frame(handler, single_handler, batch):
    backbone_input_multi, _ = handler.prepare_training_batch(batch)
    backbone_input_single, _ = single_handler.prepare_training_batch(batch[1:])
    assert torch.allclose(
        backbone_input_multi[handler.nodes]["residual_scalar"],
        backbone_input_single[single_handler.nodes]["residual_scalar"],
    )

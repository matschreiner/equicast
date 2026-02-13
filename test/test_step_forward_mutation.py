"""Tests to verify step_forward does not mutate inputs unexpectedly."""

from unittest.mock import Mock, patch

import pytest
import torch
from torch_geometric.data import HeteroData

from equicast.data.data_handler import GraphDataHandler
from equicast.data.feature_config import FeatureConfig
from equicast.model.model import Model


@pytest.fixture
def feature_config():
    """Create sample feature configuration."""
    return FeatureConfig(
        forcing=["solar", "pressure"],
        prognostic=["temp", "humidity"],
        diagnostic=["precip"],
    )


@pytest.fixture
def mock_dataset():
    """Mock anemoi dataset."""
    import numpy as np

    dataset = Mock()
    dataset.statistics = {
        "mean": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "stdev": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
    }
    dataset.name_to_index = {
        "solar": 0,
        "pressure": 1,
        "temp": 2,
        "humidity": 3,
        "precip": 4,
    }
    return dataset


@pytest.fixture
def data_handler(feature_config, mock_dataset):
    """Create a GraphDataHandler with mocked dataset."""
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        return GraphDataHandler("fake_path.zarr", feature_config, nodes="grid")


@pytest.fixture
def simple_backbone(data_handler):
    """Create a simple backbone that returns output features."""

    class SimpleBackbone(torch.nn.Module):
        def __init__(self, num_out_features):
            super().__init__()
            self.dummy_param = torch.nn.Parameter(torch.randn(1))
            self.num_out_features = num_out_features

        def forward(self, graph):
            # Return output with same shape as output features
            num_nodes = graph["grid"].input.shape[0]
            return torch.randn(num_nodes, self.num_out_features)

    return SimpleBackbone(len(data_handler.out_idxs))


@pytest.fixture
def model(simple_backbone, data_handler):
    """Create a Model instance."""
    return Model(backbone=simple_backbone, data_handler=data_handler)


@pytest.fixture
def sample_input():
    """Create a sample input graph with .data attribute."""
    graph = HeteroData()
    graph["grid"].data = torch.randn(10, 5)  # 10 nodes, 5 features
    graph["grid"].x = torch.randn(10, 2)  # positions
    return graph


@pytest.fixture
def sample_target():
    """Create a sample target graph with .data attribute."""
    graph = HeteroData()
    graph["grid"].data = torch.randn(10, 5)  # 10 nodes, 5 features
    graph["grid"].x = torch.randn(10, 2)  # positions
    return graph


def make_graph():
    """Create a graph with random data."""
    graph = HeteroData()
    graph["grid"].data = torch.randn(10, 5)  # 10 nodes, 5 features
    graph["grid"].x = torch.randn(10, 2)  # positions
    return graph


@pytest.fixture
def timeseries():
    """Create a timeseries of input/target pairs with distinct data."""
    return [
        {"input": make_graph(), "target": make_graph()},
        {"input": make_graph(), "target": make_graph()},
        {"input": make_graph(), "target": make_graph()},
    ]


class TestStepForwardMutation:
    """Tests to verify step_forward does not mutate inputs unexpectedly."""

    def test_step_forward_does_not_mutate_original_input(self, model, sample_input, sample_target):
        """Original input should be unchanged after step_forward."""
        original_data = sample_input["grid"].data.clone()

        model.step_forward(sample_input, sample_target)

        assert torch.allclose(
            sample_input["grid"].data, original_data
        ), "step_forward mutated the original input!"

    def test_step_forward_does_not_mutate_original_target(self, model, sample_input, sample_target):
        """Original target should be unchanged after step_forward."""
        original_data = sample_target["grid"].data.clone()

        model.step_forward(sample_input, sample_target)

        assert torch.allclose(
            sample_target["grid"].data, original_data
        ), "step_forward mutated the original target!"

    def test_step_forward_returns_independent_graphs(self, model, sample_input, sample_target):
        """Returned next and pred should be independent from each other."""
        next_graph, pred_graph = model.step_forward(sample_input, sample_target)

        # Mutate next_graph
        next_graph["grid"].data[:] = 999.0

        # pred_graph should be unaffected
        assert not torch.allclose(
            pred_graph["grid"].data, next_graph["grid"].data
        ), "next and pred are aliased!"

    def test_step_forward_output_independent_from_input(self, model, sample_input, sample_target):
        """Returned graphs should be independent from inputs."""
        # Save original input data BEFORE passing through the model
        original_input_data = sample_input["grid"].data.clone()
        original_target_data = sample_target["grid"].data.clone()

        next_graph, pred_graph = model.step_forward(sample_input, sample_target)

        # Mutate the original inputs
        sample_input["grid"].data[:] = -999.0
        sample_target["grid"].data[:] = -999.0

        # Outputs should NOT have been affected by input mutation
        # If they were aliased, they would now contain -999.0
        assert not torch.allclose(
            next_graph["grid"].data,
            torch.full_like(next_graph["grid"].data, -999.0),
        ), "next_graph is aliased to input!"
        assert not torch.allclose(
            pred_graph["grid"].data,
            torch.full_like(pred_graph["grid"].data, -999.0),
        ), "pred_graph is aliased to target!"

        # Verify inputs were protected (unchanged from original)
        assert torch.allclose(
            sample_input["grid"].data,
            torch.full_like(original_input_data, -999.0),
        ), "sample_input should have our mutation"
        assert torch.allclose(
            sample_target["grid"].data,
            torch.full_like(original_target_data, -999.0),
        ), "sample_target should have our mutation"

    def test_autoregressive_loop_propagates_predictions(self, model, timeseries):
        """Verify predictions actually feed back into the next step."""
        input_graph = timeseries[0]["input"].clone()
        out_idxs = model.data_handler.out_idxs

        # Get initial state
        initial_out_features = input_graph["grid"].data[..., out_idxs].clone()

        # Run one step
        next_graph, pred = model.step_forward(input_graph, timeseries[0]["target"])

        # The next_graph should have different output features (the prediction)
        next_out_features = next_graph["grid"].data[..., out_idxs]
        pred_out_features = pred["grid"].data[..., out_idxs]

        # pred's output features should match what's in next_graph
        assert torch.allclose(
            next_out_features, pred_out_features
        ), "Prediction not properly copied to next_graph!"

from unittest.mock import Mock

import pytest
import torch
from torch_geometric.data import Data, HeteroData

from equicast.data.data_handler import DataHandler
from equicast.data.feature_config import FeatureConfig
from equicast.model.model import Model


@pytest.fixture
def mock_data_handler():
    """Create a mock DataHandler."""
    handler = Mock(spec=DataHandler)

    # Mock scaler as a callable that returns data with same shape
    def scaler_side_effect(data):
        # Return data with same shape (just pass through for simplicity)
        return data

    handler.scaler = Mock(side_effect=scaler_side_effect)

    # Mock indices
    handler.in_idxs = [0, 1, 2]  # 3 input features
    handler.out_idxs = [2, 3]  # 2 output features

    # Mock the new transformation method
    def raw_to_model_side_effect(raw_state, *, target=False):
        # Scale and route to appropriate features
        idxs = handler.out_idxs if target else handler.in_idxs
        return raw_state[:, idxs]

    handler.raw_to_model = Mock(side_effect=raw_to_model_side_effect)

    return handler


@pytest.fixture
def simple_backbone():
    """Create a simple backbone for testing."""

    class SimpleBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Add a parameter so optimizer tests work
            self.dummy_param = torch.nn.Parameter(torch.randn(1))

        def forward(self, graph):
            # Simple: return the cond as prediction (ignore dummy_param)
            return graph["grid"].cond[:, :2]  # Return 2 features

    return SimpleBackbone()


@pytest.fixture
def sample_graph():
    """Create a sample graph with input and target states."""
    graph = HeteroData()
    graph["grid"].input_state = torch.randn(10, 5)  # 10 nodes, 5 features
    graph["grid"].target_state = torch.randn(10, 5)
    # Add num_graphs attribute (present in Batch objects from DataLoader)
    graph.num_graphs = 1
    return graph


def test_model_initialization(simple_backbone, mock_data_handler):
    """Test Model initialization."""
    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
    )

    assert model.backbone is simple_backbone
    assert model.data_handler is mock_data_handler
    assert model.optimizer_factory is None
    assert model.scheduler_factory is None


def test_model_initialization_with_optimizers(
    simple_backbone, mock_data_handler
):
    """Test Model initialization with optimizer factories."""
    opt_factory = lambda params: torch.optim.Adam(params, lr=0.001)
    sched_factory = lambda opt: torch.optim.lr_scheduler.StepLR(
        opt, step_size=10
    )

    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
        optimizer_factory=opt_factory,
        scheduler_factory=sched_factory,
    )

    assert model.optimizer_factory is opt_factory
    assert model.scheduler_factory is sched_factory


def test_model_forward(simple_backbone, mock_data_handler, sample_graph):
    """Test Model forward pass."""
    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
    )

    pred = model.forward(sample_graph)

    # Should call raw_to_model on input_state with target=False
    mock_data_handler.raw_to_model.assert_called()
    assert mock_data_handler.raw_to_model.call_args.kwargs["target"] == False

    # Should return a prediction
    assert pred is not None
    assert isinstance(pred, torch.Tensor)


def test_model_forward_processes_input_only(
    simple_backbone, mock_data_handler, sample_graph
):
    """Test that forward only processes input, not target."""
    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
    )

    pred = model.forward(sample_graph)

    # Should only call prepare_model_input once (for input)
    assert mock_data_handler.prepare_model_input.call_count == 1

    # Should be called with input_state
    call_args = mock_data_handler.prepare_model_input.call_args[0][0]
    assert torch.equal(call_args, sample_graph["grid"].input_state)


def test_model_training_step(simple_backbone, mock_data_handler, sample_graph):
    """Test Model training_step."""
    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
    )

    # Mock scaler to return scaled tensors
    def scaler_side_effect(data):
        return data  # Just return same shape for simplicity

    mock_data_handler.scaler = Mock(side_effect=scaler_side_effect)

    loss = model.training_step(sample_graph, 0)

    # Should return a scalar loss
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0  # Loss should be non-negative


def test_model_training_step_processes_target(
    simple_backbone, mock_data_handler, sample_graph
):
    """Test that training_step processes both input and target."""
    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
    )

    model.training_step(sample_graph, 0)

    # Should call prepare_model_input once (in forward) and prepare_model_target once
    assert mock_data_handler.prepare_model_input.call_count == 1
    assert mock_data_handler.prepare_model_target.call_count == 1


def test_model_configure_optimizers_default(simple_backbone, mock_data_handler):
    """Test default optimizer configuration."""
    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
    )

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert isinstance(optimizer, torch.optim.Adam)


def test_model_configure_optimizers_custom(simple_backbone, mock_data_handler):
    """Test custom optimizer configuration."""
    opt_factory = lambda params: torch.optim.SGD(params, lr=0.01)

    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
        optimizer_factory=opt_factory,
    )

    optimizer = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.SGD)


def test_model_configure_optimizers_with_scheduler(
    simple_backbone, mock_data_handler
):
    """Test optimizer configuration with scheduler."""
    opt_factory = lambda params: torch.optim.Adam(params, lr=0.001)
    sched_factory = lambda opt: torch.optim.lr_scheduler.StepLR(
        opt, step_size=10
    )

    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
        optimizer_factory=opt_factory,
        scheduler_factory=sched_factory,
    )

    result = model.configure_optimizers()

    # Should return [optimizers], [schedulers]
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], list)
    assert isinstance(result[1], list)
    assert isinstance(result[0][0], torch.optim.Optimizer)
    assert isinstance(result[1][0], torch.optim.lr_scheduler.LRScheduler)


def test_model_is_lightning_module(simple_backbone, mock_data_handler):
    """Test that Model is a PyTorch Lightning module."""
    import pytorch_lightning as pl

    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
    )

    assert isinstance(model, pl.LightningModule)


def test_model_saves_hyperparameters(simple_backbone, mock_data_handler):
    """Test that Model saves hyperparameters."""
    model = Model(
        backbone=simple_backbone,
        data_handler=mock_data_handler,
    )

    # Should have hparams saved
    assert hasattr(model, "hparams")

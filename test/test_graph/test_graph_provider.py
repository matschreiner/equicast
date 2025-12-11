import pytest
import tempfile
import torch
from pathlib import Path
from torch_geometric.data import Data, HeteroData

from equicast.graph.graph_provider import BaseGraphProvider, StaticGraphProvider


@pytest.fixture
def sample_graph():
    """Create a sample PyTorch Geometric graph."""
    return HeteroData({
        "grid": Data(x=torch.randn(100, 10))
    })


@pytest.fixture
def graph_file(sample_graph):
    """Create a temporary file with a saved graph."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(sample_graph, f.name)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


def test_static_graph_provider_initialization(graph_file):
    """Test StaticGraphProvider initialization."""
    provider = StaticGraphProvider(graph_file)

    assert provider.graph is not None
    assert isinstance(provider.graph, (Data, HeteroData))


def test_static_graph_provider_get_graph(graph_file):
    """Test getting graph from StaticGraphProvider."""
    provider = StaticGraphProvider(graph_file)

    graph = provider.get_graph()

    assert graph is not None
    assert isinstance(graph, (Data, HeteroData))


def test_static_graph_provider_returns_clone(graph_file):
    """Test that get_graph returns a clone, not the original."""
    provider = StaticGraphProvider(graph_file)

    graph1 = provider.get_graph()
    graph2 = provider.get_graph()

    # Should be different objects (clones)
    assert graph1 is not graph2
    assert graph1 is not provider.graph


def test_static_graph_provider_clone_has_same_data(graph_file):
    """Test that cloned graph has same data as original."""
    provider = StaticGraphProvider(graph_file)
    original = provider.graph

    cloned = provider.get_graph()

    # Data should be equal
    assert torch.equal(cloned["grid"].x, original["grid"].x)


def test_static_graph_provider_idx_parameter_ignored(graph_file):
    """Test that idx parameter is ignored for StaticGraphProvider."""
    provider = StaticGraphProvider(graph_file)

    graph1 = provider.get_graph(idx=0)
    graph2 = provider.get_graph(idx=100)

    # Should return same graph regardless of idx
    assert torch.equal(graph1["grid"].x, graph2["grid"].x)


def test_static_graph_provider_to_device(graph_file):
    """Test moving StaticGraphProvider to device."""
    provider = StaticGraphProvider(graph_file)

    # Move to CPU (should work even if already on CPU)
    provider_cpu = provider.to('cpu')

    assert provider_cpu is provider  # Should return self
    assert provider.graph["grid"].x.device.type == 'cpu'


def test_static_graph_provider_is_base_graph_provider(graph_file):
    """Test that StaticGraphProvider is a BaseGraphProvider."""
    provider = StaticGraphProvider(graph_file)

    assert isinstance(provider, BaseGraphProvider)


def test_static_graph_provider_is_nn_module(graph_file):
    """Test that StaticGraphProvider is a torch.nn.Module."""
    provider = StaticGraphProvider(graph_file)

    assert isinstance(provider, torch.nn.Module)


def test_base_graph_provider_is_abstract():
    """Test that BaseGraphProvider cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseGraphProvider()


def test_static_graph_provider_with_simple_data():
    """Test StaticGraphProvider with simple homogeneous graph."""
    # Create simple graph
    simple_graph = Data(x=torch.randn(10, 5), edge_index=torch.randint(0, 10, (2, 20)))

    # Save and load
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(simple_graph, f.name)
        temp_path = f.name

    try:
        provider = StaticGraphProvider(temp_path)
        loaded_graph = provider.get_graph()

        assert isinstance(loaded_graph, Data)
        assert torch.equal(loaded_graph.x, simple_graph.x)
        assert torch.equal(loaded_graph.edge_index, simple_graph.edge_index)
    finally:
        Path(temp_path).unlink()

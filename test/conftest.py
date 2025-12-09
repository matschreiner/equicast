import pytest
from torch_geometric.loader.dataloader import DataLoader

from equicast.dataset import AnemoiDataset
from equicast.graph.graph_provider import StaticGraphProvider


@pytest.fixture
def dataset(graph_provider):
    return AnemoiDataset(
        path="test/res/micro_aifs.zarr",
        graph_provider=graph_provider,
    )


@pytest.fixture
def batch(dataset):
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)

    for batch in dataloader:
        return batch


@pytest.fixture
def graph_provider():
    return StaticGraphProvider(
        path="test/res/micro_aifs.pt",
    )


@pytest.fixture
def features():
    return {
        "forcing": ["lsm", "cos_julian_day"],
        "prognostic": ["10u"],
        "diagnostic": ["msl"],
    }


@pytest.fixture
def graph(graph_provider):
    return graph_provider.get_graph(0)

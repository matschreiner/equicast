from torch_geometric.data import HeteroData
from torch_geometric.loader.dataloader import DataLoader


def test_dataset(dataset):
    graph = dataset[0]
    assert isinstance(graph, HeteroData)


def test_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    assert isinstance(batch, HeteroData)
    assert len(batch.to_data_list()) == 2

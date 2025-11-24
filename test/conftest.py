import pytest
from torch.utils.data import DataLoader

from equicast.dataset import AnemoiDataset


@pytest.fixture
def dataset():
    return AnemoiDataset(
        path="test/res/micro_aifs.zarr",
    )


@pytest.fixture
def batch(dataset):
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for batch in dataloader:
        return batch

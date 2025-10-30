from anemoi.datasets import open_dataset
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, path):
        self.data = open_dataset(path)
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

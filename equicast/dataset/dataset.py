import torch
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from torch.utils.data import Dataset

from equicast import DTYPE
from equicast.data.feature_router import FeatureRouter
from equicast.data.scaler import Scaler
from equicast.utils import utils


class AnemoiDataset(Dataset):
    def __init__(self, path, graph_provider):
        super().__init__()
        self.data = open_dataset(path)
        self.statistics = utils.cast_dict(self.data.statistics, torch.Tensor)
        self.name_to_index = self.data.name_to_index
        self.graph_provider = graph_provider
        self.scaler = Scaler(self.statistics)
        self.data = self.data[:10]

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        graph = self.graph_provider.get_graph(idx)

        raw = torch.Tensor(self.data[idx : idx + 2]).squeeze().permute([2, 0, 1])
        graph["data"].raw = raw
        return graph

import torch
from anemoi.datasets import open_dataset
from anemoi.utils.config import DotDict
from torch.utils.data import Dataset

from equicast import DTYPE
from equicast.utils import utils


class AnemoiDataset(Dataset):
    def __init__(self, path, graph_provider=None):
        super().__init__()
        self.data = open_dataset(path)
        self.statistics = utils.cast_dict(self.data.statistics, torch.Tensor)
        self.graph_provider = graph_provider

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        data = self.data[idx : idx + 2].squeeze()
        data = data.transpose([0, 2, 1])
        data = torch.tensor(data, dtype=DTYPE)

        batch = {"data": data, "idx": idx, "name_to_idx": self.data.name_to_index}

        if self.graph_provider is not None:
            graph = self.graph_provider.get_graph(idx)
            batch["graph"] = graph

        return DotDict(batch)

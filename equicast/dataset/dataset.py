import torch
from anemoi.datasets import open_dataset
from torch.utils.data import Dataset

from equicast.data.scaler import Scaler
from equicast.utils import utils


class AnemoiDataset(Dataset):
    def __init__(self, path, graph_provider, features):
        super().__init__()
        self.data = open_dataset(path)
        self.statistics = utils.cast_dict(self.data.statistics, torch.Tensor)
        self.name_to_index = self.data.name_to_index
        self.graph_provider = graph_provider
        self.scaler = Scaler(self.statistics)

        self.cond_idxs = self._get_idxs(features.forcing) + self._get_idxs(
            features.prognostic
        )
        self.target_idxs = self._get_idxs(features.prognostic) + self._get_idxs(
            features.diagnostic
        )

    def __len__(self):
        return len(self.data) - 1

    def _get_idxs(self, names):
        return [self.name_to_index[name] for name in names]

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx : idx + 2]).squeeze().permute(0, 2, 1)
        data = self.scaler(data)

        cond = data[0][:, self.cond_idxs]
        target = data[1][:, self.target_idxs]

        graph = self.graph_provider.get_graph()

        graph["data"].cond = cond
        graph["data"].target = target

        return graph

import torch
from anemoi.datasets import open_dataset
from torch.utils.data import Dataset

from equicast import DTYPE


class AnemoiDataset(Dataset):
    def __init__(self, path, variables, graph_provider=None):
        super().__init__()
        self.data = open_dataset(path)
        self.data.name_to_index
        self.forcing_idxs = self._get_data_idxs(variables.forcing)
        self.prognostic_idxs = self._get_data_idxs(variables.prognostic)
        self.diagnostic_idxs = self._get_data_idxs(variables.diagnostic)
        self.statistics = self.to_torch(self.data.statistics)
        self.graph_provider = graph_provider

    def _get_data_idxs(self, names):
        return [self.data.name_to_index[name] for name in names]

    def __len__(self):
        return len(self.data) - 1

    def to_torch(self, statistics):
        for key in statistics:
            statistics[key] = torch.tensor(statistics[key], dtype=DTYPE)
        return statistics

    def normalize(self, data):
        mean = self.statistics["mean"]
        std = self.statistics["stdev"]
        return (data - mean) / std

    def __getitem__(self, idx):
        cond = torch.tensor(self.data[idx].squeeze()).T
        target = torch.tensor(self.data[idx + 1].squeeze()).T

        cond = self.normalize(cond)
        target = self.normalize(target)

        forcing = cond[self.forcing_idxs]
        prognostic = cond[self.prognostic_idxs]
        cond = torch.concatenate([forcing, prognostic])

        prognostic = target[self.prognostic_idxs]
        diagnostic = target[self.diagnostic_idxs]
        target = torch.concatenate([prognostic, diagnostic])

        batch = {"condition": cond.T, "target": target.T, "idx": idx}

        if self.graph_provider is not None:
            graph = self.graph_provider.get_graph(idx)
            batch["graph"] = graph

        return batch

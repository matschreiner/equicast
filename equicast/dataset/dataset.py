import torch
from anemoi.datasets import open_dataset
from torch.utils.data import Dataset

from equicast.data.data_handler import DataHandler


class AnemoiDataset(Dataset):
    def __init__(self, path, graph_provider, data_handler: DataHandler):
        super().__init__()
        self.data = open_dataset(path)
        self.graph_provider = graph_provider
        self.data_handler = data_handler

        # Use shared scaler and indices from data_handler
        self.scaler = data_handler.scaler
        self.in_idxs = data_handler.in_idxs
        self.out_idxs = data_handler.out_idxs

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx : idx + 2]).squeeze().permute(0, 2, 1)
        data = self.scaler(data)

        cond = data[0][:, self.in_idxs]
        target = data[1][:, self.out_idxs]

        graph = self.graph_provider.get_graph()

        graph["grid"].cond = cond
        graph["grid"].target = target

        return graph

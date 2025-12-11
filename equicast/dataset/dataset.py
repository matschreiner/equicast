import torch
from anemoi.datasets import open_dataset
from torch.utils.data import Dataset

from equicast.graph.graph_provider import BaseGraphProvider


class AnemoiDataset(Dataset):
    """
    Dataset that loads raw weather data without preprocessing.

    Preprocessing (scaling, feature routing) is handled by the Model.
    """

    def __init__(self, path: str, graph_provider: BaseGraphProvider):
        super().__init__()
        self.data = open_dataset(path)
        self.graph_provider = graph_provider

    def __len__(self):
        return len(self.data) - 1

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx : idx + 2]).squeeze().permute(0, 2, 1)

        graph = self.graph_provider.get_graph(idx)

        graph["grid"].input_state = data[0]
        graph["grid"].target_state = data[1]

        return graph

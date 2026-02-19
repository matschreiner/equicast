import torch
from anemoi.datasets import open_dataset
from torch.utils.data import Dataset

from equicast.data.graph_provider import BaseGraphProvider


class AnemoiDataset(Dataset):
    """
    Dataset that loads raw weather data without preprocessing.

    Preprocessing (scaling, feature routing) is handled by the Model.
    """

    def __init__(
        self,
        path: str,
        graph_provider: BaseGraphProvider,
        step: int = 1,
    ):
        super().__init__()
        self.data = open_dataset(path)
        self.graph_provider = graph_provider
        self.step = step

    def __len__(self):
        return len(self.data) - self.step

    def __getitem__(self, idx):
        input_idx = idx
        target_idx = idx + self.step
        input_data = self.data[input_idx].squeeze()
        target_data = self.data[target_idx].squeeze()

        input_graph = self.graph_provider.get_graph(idx)
        target_graph = self.graph_provider.get_graph(idx)

        input_graph["grid"].data = torch.from_numpy(input_data.T)
        target_graph["grid"].data = torch.from_numpy(target_data.T)

        return {"input": input_graph, "target": target_graph}

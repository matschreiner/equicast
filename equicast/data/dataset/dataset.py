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
        subsample: int = 1,
    ):
        super().__init__()
        self.data = open_dataset(path)
        self.graph_provider = graph_provider
        self.subsample = subsample

    def __len__(self):
        return (len(self.data) - self.subsample) // self.subsample

    def __getitem__(self, idx):
        input_idx = idx * self.subsample
        target_idx = input_idx + self.subsample
        input_data = self.data[input_idx].squeeze()
        target_data = self.data[target_idx].squeeze()

        input_graph = self.graph_provider.get_graph(idx)
        target_graph = self.graph_provider.get_graph(idx)

        #  graph_path = "graph/aifs-graphcast-unnormed.pt"
        #  input_graph = torch.load(graph_path, weights_only=False)
        #  target_graph = torch.load(graph_path, weights_only=False)

        input_graph["grid"].data = torch.from_numpy(input_data).T

        target_graph["grid"].data = torch.from_numpy(target_data).T

        return {"input": input_graph, "target": target_graph}

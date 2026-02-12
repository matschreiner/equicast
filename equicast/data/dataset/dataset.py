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
        num_input_steps: int = 1,
    ):
        super().__init__()
        self.data = open_dataset(path)
        self.graph_provider = graph_provider
        self.num_input_steps = num_input_steps

    def __len__(self):
        return len(self.data) - self.num_input_steps

    def __getitem__(self, idx):
        data = (
            torch.tensor(self.data[idx : idx + self.num_input_steps + 1]).squeeze().permute(0, 2, 1)
        )

        input_graph = self.graph_provider.get_graph(idx)
        input_graph["grid"].data = data[:-1].squeeze()

        target_graph = self.graph_provider.get_graph(idx)
        target_graph["grid"].data = data[-1:].squeeze()

        return {"input": input_graph, "target": target_graph}

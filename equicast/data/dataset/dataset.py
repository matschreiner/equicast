import torch
from anemoi.datasets import open_dataset
from torch.utils.data import Dataset

from equicast.data.graph_provider import BaseGraphProvider


class AnemoiDataset(Dataset):
    """
    Dataset that loads raw weather data without preprocessing.

    Returns no_frames consecutive frames spaced by step timesteps.
    Preprocessing (scaling, feature routing) is handled by the Model.
    """

    def __init__(
        self,
        path: str,
        graph_provider: BaseGraphProvider,
        no_frames: int = 2,
        step: int = 1,
        node_name="data",
        variables: list[str] | None = None,
    ):
        super().__init__()
        self.data = open_dataset(path)
        self.graph_provider = graph_provider
        self.no_frames = no_frames
        self.step = step
        self.node_name = node_name
        self._var_indices = [self.data.name_to_index[v] for v in variables] if variables is not None else None

    def __len__(self):
        return len(self.data) - (self.no_frames - 1) * self.step

    def __getitem__(self, idx):
        frames = []

        for i in range(self.no_frames):
            frame_idx = idx + i * self.step
            raw = self.data[frame_idx].squeeze()
            if self._var_indices is not None:
                raw = raw[self._var_indices]
            graph = self.graph_provider.get_graph(idx)
            graph[self.node_name].data = torch.from_numpy(raw.T)
            frames.append(graph)

        return frames

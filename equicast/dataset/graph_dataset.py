from torch.utils.data import Dataset

#  from equicast import DTYPE
#  from equicast.graphs.static_graph_provider import BaseGraphProvider


class GraphDataset(Dataset):
    def __init__(self, dataset: Dataset, graph_provider):
        super().__init__()
        self.dataset = dataset
        self.graph_provider = graph_provider

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        graph = self.graph_provider.get_graph(idx)
        batch["graph"] = graph
        return batch

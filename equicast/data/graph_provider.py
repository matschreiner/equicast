from abc import ABC, abstractmethod

import torch
from torch_geometric.data import HeteroData


class BaseGraphProvider(ABC, torch.nn.Module):
    @abstractmethod
    def get_graph(self, idx=None) -> HeteroData: ...


class StaticGraphProvider(BaseGraphProvider):
    def __init__(self, graph_path):
        super().__init__()
        self.graph = torch.load(graph_path, weights_only=False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.graph = self.graph.to(*args, **kwargs)
        return self

    def get_graph(self, idx=None) -> HeteroData:
        return self.graph.clone()

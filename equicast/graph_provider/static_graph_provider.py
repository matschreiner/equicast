from abc import ABC, abstractmethod

import torch


class BaseGraphProvider(ABC, torch.nn.Module):
    @abstractmethod
    def get_graph(self, idx): ...


class StaticGraphProvider(BaseGraphProvider):
    def __init__(self, path):
        super().__init__()
        self.graph = torch.load(path, weights_only=False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.graph = self.graph.to(*args, **kwargs)
        return self

    def get_graph(self, idx):
        return self.graph

import torch

from equicast.graph.graph_provider import BaseGraphProvider


class Forecaster:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def forecast(self, batch, steps):
        return

import torch


class Forecaster(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch: dict, steps) -> torch.Tensor:
        pred = self.model(batch)

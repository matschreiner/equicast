import torch


class Forecaster:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def forecast(self, batch, steps):
        return

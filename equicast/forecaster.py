import torch


class Forecaster(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, scaler: callable = None):
        super().__init__()
        self.model = model
        self.scaler = scaler

    def forward(self, batch: dict) -> torch.Tensor:
        pred = self.model(batch)
        if self.scaler is not None:
            pred = self.scaler.inverse_transform(pred)
        return pred

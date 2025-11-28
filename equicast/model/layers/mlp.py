import torch
from anemoi.models.layers.normalization import AutocastLayerNorm


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        activation=torch.nn.GELU(),
        norm=AutocastLayerNorm,
        num_layers=3,
        hidden_dim=None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = out_dim

        layers = []
        for i in range(num_layers):
            in_dim = in_dim if i == 0 else hidden_dim
            out_dim = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(torch.nn.Linear(in_dim, out_dim))
            if i != num_layers - 1:
                layers.append(norm(out_dim))
                layers.append(activation)
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

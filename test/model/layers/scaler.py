import torch

from equicast.model.layers.scaler import Scaler


def test_scaler_instantiate(dataset, batch):
    scaler = Scaler(dataset.statistics)
    scaled_data = scaler.transform(batch["data"])
    data = scaler.inverse_transform(scaled_data)

    assert torch.allclose(data, batch["data"])

# test_simple_data_processor.py

import pytest
import torch

from equicast.data.processor import SimpleDataProcessor  # adjust import


@pytest.fixture
def processor(dataset, features):
    return SimpleDataProcessor(
        statistics=dataset.statistics,
        name_to_index=dataset.name_to_index,
        features=features,
    )


def test_route_features_slices_correctly(processor, batch):
    data = batch["data"]
    raw = data.raw.clone()

    data = processor.route_features(data)

    in_idxs = processor.in_idxs
    out_idxs = processor.out_idxs

    assert data.cond.shape[-1] == len(in_idxs)
    assert data.target.shape[-1] == len(out_idxs)

    expected_cond = raw[:, 0, in_idxs]
    expected_target = raw[:, 1, out_idxs]

    assert torch.allclose(data.cond, expected_cond)
    assert torch.allclose(data.target, expected_target)


def test_scale_data_scales_only_routed_features(processor, batch, dataset):
    data = batch["data"]
    data = processor.route_features(data)

    mean = dataset.statistics["mean"]
    std = dataset.statistics["stdev"]

    cond_before = data.cond.clone()
    target_before = data.target.clone()

    data = processor.scale_data(data)

    mean_in = mean[..., processor.in_idxs]
    std_in = std[..., processor.in_idxs]
    mean_out = mean[..., processor.out_idxs]
    std_out = std[..., processor.out_idxs]

    expected_cond = (cond_before - mean_in) / std_in
    expected_target = (target_before - mean_out) / std_out

    assert torch.allclose(data.cond, expected_cond)
    assert torch.allclose(data.target, expected_target)


def test_inverse_scale_restores_routed_space(processor, batch, dataset):
    data = batch["data"]
    data = processor.route_features(data)
    data = processor.scale_data(data)

    # pretend model output is the scaled target
    data.pred = data.target.clone()

    data = processor.inverse_scale_data(data)

    mean = dataset.statistics["mean"]
    std = dataset.statistics["stdev"]
    target_physical = data.raw[:, 1, processor.out_idxs]

    mean_out = mean[..., processor.out_idxs]
    std_out = std[..., processor.out_idxs]

    expected_target_physical = (data.target * std_out) + mean_out

    assert torch.allclose(data.pred, expected_target_physical)
    assert torch.allclose(data.pred, target_physical)


def test_inverse_route_reconstructs_full_field(processor, batch):
    data = batch["data"]
    raw = data.raw.clone()

    data = processor.route_features(data)
    data = processor.scale_data(data)

    data.pred = data.target.clone()

    data = processor.inverse_scale_data(data)
    data = processor.inverse_route_features(data)

    reconstructed = data.reconstructed

    assert reconstructed.shape == raw.shape

    mask = torch.ones(raw.shape[-1], dtype=torch.bool, device=raw.device)
    mask[processor.out_idxs] = False

    # unchanged features
    assert torch.allclose(reconstructed[:, 1, mask], raw[:, 1, mask])
    # routed features restored to original after inverse operations
    assert torch.allclose(
        reconstructed[:, 1, processor.out_idxs],
        raw[:, 1, processor.out_idxs],
    )

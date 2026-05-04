import pytorch_lightning as pl
import torch
import pytest
from unittest.mock import Mock, patch
from torch_geometric.loader.dataloader import DataLoader

from equicast.callbacks.checkpoint import CheckpointSaver
from equicast.data import AnemoiDataset
from equicast.data.data_handler import GraphDataHandler
from equicast.data.feature_config import FeatureConfig
from equicast.data.graph_provider import StaticGraphProvider
from equicast.model.deterministic import Deterministic
from equicast.model.backbones.gnn import GNN

ZARR_PATH = "test/res/micro_aifs.zarr"
GRAPH_PATH = "test/res/micro_aifs.pt"
FEATURES = FeatureConfig(forcing=["lsm", "cos_julian_day"], prognostic=["10u"], diagnostic=["msl"])


@pytest.fixture
def feature_config():
    return FeatureConfig(
        forcing=["f1"],
        prognostic=["p1", "p2"],
        diagnostic=[],
    )


@pytest.fixture
def mock_dataset():
    import numpy as np
    dataset = Mock()
    dataset.statistics = {
        "mean": np.zeros(3),
        "stdev": np.ones(3),
    }
    dataset.name_to_index = {"f1": 0, "p1": 1, "p2": 2}
    return dataset


@pytest.fixture
def data_handler(feature_config, mock_dataset):
    with patch("equicast.data.data_handler.open_dataset", return_value=mock_dataset):
        return GraphDataHandler("fake.zarr", feature_config, nodes="data")


def make_model(data_handler):
    backbone = GNN(
        in_dim=data_handler.in_dim,
        out_dim=data_handler.out_dim,
        edges=[("data", "to", "data")],
        hidden_dim=8,
    )
    return Deterministic(backbone=backbone, data_handler=data_handler, compile_backbone=False)


def test_checkpoint_preserves_weights(data_handler, tmp_path):
    model = make_model(data_handler)

    trainer = Mock()
    trainer.is_global_zero = True
    trainer.save_checkpoint = lambda path: torch.save({"state_dict": model.state_dict()}, path)

    saver = CheckpointSaver(dirpath=str(tmp_path))
    saver.save(trainer, "model.ckpt")

    model2 = make_model(data_handler)
    ckpt = torch.load(tmp_path / "model.ckpt", weights_only=False)
    model2.load_state_dict(ckpt["state_dict"])

    for (name, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), f"Mismatch in {name}"


def make_real_model():
    dh = GraphDataHandler(ZARR_PATH, FEATURES, nodes="data")
    backbone = GNN(
        in_dim=dh.in_dim,
        out_dim=dh.out_dim,
        edges=[("data", "to", "data")],
        hidden_dim=8,
    )
    return Deterministic(backbone=backbone, data_handler=dh, compile_backbone=False)


def make_dataloader():
    gp = StaticGraphProvider(GRAPH_PATH)
    ds = AnemoiDataset(ZARR_PATH, graph_provider=gp)
    return DataLoader(ds, batch_size=2, shuffle=False)


def test_pl_checkpoint_preserves_weights(tmp_path):
    model = make_real_model()
    dl = make_dataloader()

    trainer = pl.Trainer(max_steps=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, dl)
    trainer.save_checkpoint(tmp_path / "model.ckpt")

    model2 = make_real_model()
    trainer2 = pl.Trainer(max_steps=0, logger=False, enable_checkpointing=False)
    trainer2.fit(model2, dl, ckpt_path=tmp_path / "model.ckpt")

    for (name, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), f"Mismatch in {name}"

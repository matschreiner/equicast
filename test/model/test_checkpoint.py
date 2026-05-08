import pytorch_lightning as pl
import torch
from torch_geometric.loader.dataloader import DataLoader

from equicast.data import AnemoiDataset
from equicast.data.data_handler import GraphDataHandler
from equicast.data.feature_config import FeatureConfig
from equicast.data.graph_provider import StaticGraphProvider
from equicast.model.deterministic import Deterministic
from equicast.model.backbones.gnn import GNN

ZARR_PATH = "test/res/micro_aifs.zarr"
GRAPH_PATH = "test/res/micro_aifs.pt"
FEATURES = FeatureConfig(forcing=["lsm", "cos_julian_day"], prognostic=["10u"], diagnostic=["msl"])


def make_model():
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
    model = make_model()
    dl = make_dataloader()

    trainer = pl.Trainer(max_steps=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, dl)
    trainer.save_checkpoint(tmp_path / "model.ckpt")

    model2 = make_model()
    trainer2 = pl.Trainer(max_steps=0, logger=False, enable_checkpointing=False)
    trainer2.fit(model2, dl, ckpt_path=tmp_path / "model.ckpt")

    for (name, p1), (_, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), f"Mismatch in {name}"

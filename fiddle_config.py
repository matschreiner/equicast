from dataclasses import dataclass
from functools import partial

import fiddle as fdl
import torch
from fiddle import graphviz
from fiddle.experimental import auto_config
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch_geometric.loader import DataLoader

from equicast.dataset import AnemoiDataset
from equicast.graph.graph_provider import StaticGraphProvider
from equicast.logger.mlflow import MLFlowLogger
from equicast.model.backbones.simple import Simple
from equicast.model.model import Model


@dataclass
class FeatureConfig:
    forcing: list[str]
    prognostic: list[str]
    diagnostic: list[str]

    @classmethod
    def from_yaml(cls, path: str):
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


@auto_config.auto_config
def experiment_config():
    feature_config = FeatureConfig.from_yaml("config/features/base.yaml")

    dataset = AnemoiDataset(
        path="/home/masc/storage/mini_aifs.zarr",
        features=feature_config,
        graph_provider=StaticGraphProvider(
            path="./graph/aifs-single.pt",
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )

    backbone = Simple(feature_config)
    optimizer_factory = partial(Adam, lr=1e-3)
    scheduler_factory = torch.optim.lr_scheduler.StepLR

    model = Model(backbone=backbone, optimizer_factory=optimizer_factory)

    logger = MLFlowLogger(
        experiment_name="masc",
        tracking_uri="https://mlflow.dmidev.org/",
    )

    trainer = Trainer(
        logger=logger,
    )

    return model, trainer, dataloader, logger


def vis_graph(config):
    graph = graphviz.render(config)
    graph.view()


def main():
    config = experiment_config.as_buildable()
    vis_graph(config)
    model, trainer, dataloader, logger = fdl.build(config)

    trainer.fit(model, dataloader)


main()

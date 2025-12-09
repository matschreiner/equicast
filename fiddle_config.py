import fiddle as fdl
from fiddle import graphviz
from pytorch_lightning import Trainer
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from equicast.data import FeatureConfig
from equicast.dataset import AnemoiDataset
from equicast.graph.graph_provider import StaticGraphProvider
from equicast.logger.mlflow import MLFlowLogger
from equicast.model.backbones.simple import Simple
from equicast.model.model import Model


def make_experiment_config():
    feature_config = FeatureConfig.from_yaml("config/features/base.yaml")

    dataset = fdl.Config(
        AnemoiDataset,
        path="/home/masc/storage/mini_aifs.zarr",
        features=feature_config,
        graph_provider=StaticGraphProvider(
            path="./graph/aifs-single.pt",
        ),
    )

    dataloader = fdl.Config(
        DataLoader,
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )

    backbone = fdl.Config(
        Simple,
        feature_config,
    )

    optimizer_factory = fdl.Partial(
        Adam,
        lr=1e-3,
    )

    scheduler_factory = fdl.Partial(
        StepLR,
        step_size=10,
        gamma=0.1,
    )

    model = fdl.Config(
        Model,
        backbone=backbone,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )

    logger = fdl.Config(
        MLFlowLogger,
        experiment_name="masc",
        tracking_uri="https://mlflow.dmidev.org/",
    )

    trainer = fdl.Config(
        Trainer,
        logger=logger,
    )

    return model, trainer, dataloader, logger


def vis_graph(config):
    graph = graphviz.render(config)
    graph.view()


def main():
    config = make_experiment_config()
    vis_graph(config)
    model, trainer, dataloader, logger = fdl.build(config)
    trainer.fit(model, dataloader)


main()

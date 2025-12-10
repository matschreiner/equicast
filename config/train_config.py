import fiddle as fdl
import torch
from fiddle import graphviz
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from equicast import utils
from equicast.data import FeatureConfig
from equicast.dataset import AnemoiDataset
from equicast.graph.graph_provider import StaticGraphProvider
from equicast.logger.mlflow import MLFlowLogger
from equicast.model.backbones.simple import Simple
from equicast.model.model import Model

torch.set_float32_matmul_precision("medium | high")

from dataclasses import dataclass


def make_experiment_config():
    feature_config = FeatureConfig.from_yaml("hydraconfig/features/base.yaml")

    graph_provider = fdl.Config(
        StaticGraphProvider,
        path="./graph/aifs-single.pt",
    )

    dataset = fdl.Config(
        AnemoiDataset,
        path="/home/masc/storage/mini_aifs.zarr",
        feature_config=feature_config,
        graph_provider=graph_provider,
    )

    dataloader = fdl.Config(
        DataLoader,
        dataset,
        batch_size=1,
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

    logger = fdl.Config(
        MLFlowLogger,
        experiment_name="masc",
        tracking_uri="https://mlflow.dmidev.org/",
    )

    model = fdl.Config(
        Model,
        backbone=backbone,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )

    trainer = fdl.Config(
        Trainer,
        logger=logger,
        callbacks=[
            fdl.Config(
                ModelCheckpoint,
                every_n_epochs=1,
                save_top_k=1,
                filename="latest",
            ),
            fdl.Config(
                ModelCheckpoint,
                every_n_epochs=1,
                save_top_k=1,
                mode="min",
                monitor="loss",
                filename="minloss",
            ),
        ],
    )

    experiment = fdl.Config(
        Experiment,
        model,
        trainer,
        dataloader,
        logger,
    )

    return experiment


@dataclass
class Experiment:
    model: Model
    trainer: Trainer
    dataloader: DataLoader
    logger: Logger

    def run(self):
        self.trainer.fit(
            self.model,
            self.dataloader,
        )


def main():
    config = make_experiment_config()

    import pickle as pkl

    with open("train_config.pkl", "wb") as f:
        pkl.dump(config, f)

    utils.vis_config(config)
    experiment = fdl.build(config)
    #  experiment.run()


main()

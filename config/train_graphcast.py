"""Training configuration for GraphCast experiments."""

import fiddle as fdl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from config.base_config import get_dataset
from equicast import experiments
from equicast.data.data_handler import DataHandler
from equicast.data.feature_config import FeatureConfig
from equicast.experiments import TrainConfig
from equicast.graph.graph_provider import StaticGraphProvider
from equicast.logger.mlflow import MLFlowLogger
from equicast.model.backbones.encprocdec import EncProcDec
from equicast.model.model import Model

torch.set_float32_matmul_precision("medium")


def main():
    graph_provider = fdl.Config(
        StaticGraphProvider, path="./graph/aifs-graphcast.pt"
    )
    dataset = get_dataset(graph_provider=graph_provider)
    feature_config = fdl.Config(
        FeatureConfig.from_yaml, path="hydraconfig/features/base.yaml"
    )
    dataset_path = "/home/masc/storage/mini_aifs.zarr"

    dataloader = fdl.Config(
        DataLoader,
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=10,
    )

    data_handler = fdl.Config(
        DataHandler,
        dataset_path=dataset_path,
        feature_config=feature_config,
    )

    # Optimizer and scheduler
    optimizer_factory = fdl.Partial(
        Adam,
        lr=1e-3,
    )

    scheduler_factory = fdl.Partial(
        StepLR,
        step_size=10,
        gamma=0.1,
    )

    # Logger
    logger = fdl.Config(
        MLFlowLogger,
        experiment_name="masc",
        tracking_uri="https://mlflow.dmidev.org/",
    )

    backbone = fdl.Config(
        EncProcDec,
        feature_config=feature_config,
    )

    # Model
    model = fdl.Config(
        Model,
        backbone=backbone,
        data_handler=data_handler,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
    )

    # Trainer
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

    # Experiment config
    experiment_cfg = fdl.Config(
        TrainConfig,
        model,
        trainer,
        dataloader,
        logger,
    )

    experiments.run_experiment(experiment_cfg)


if __name__ == "__main__":
    main()

import fiddle as fdl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from config.base_config import (
    get_data_handler,
    get_dataset,
    get_feature_config,
    get_graph_provider,
)
from equicast import experiments
from equicast.experiments import TrainConfig
from equicast.logger.mlflow import MLFlowLogger
from equicast.model.backbones.gnn import GNN
from equicast.model.model import Model

torch.set_float32_matmul_precision("medium | high")


def main():
    # Get shared base configurations
    feature_config = get_feature_config()
    data_handler = get_data_handler(feature_config=feature_config)
    dataset = get_dataset()

    dataloader = fdl.Config(
        DataLoader,
        dataset,
        batch_size=1,
        shuffle=True,
        #  num_workers=4,
    )

    backbone = fdl.Config(
        GNN,
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
        data_handler=data_handler,
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

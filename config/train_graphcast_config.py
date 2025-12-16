import fiddle as fdl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader

from equicast import data, experiments
from equicast.experiments import TrainConfig
from equicast.logger.mlflow import MLFlowLogger
from equicast.model.backbones.graphcast import Graphcast
from equicast.model.model import Model

torch.set_float32_matmul_precision("medium | high")


def main():
    dataset_path = "/home/masc/storage/mini_aifs.zarr"
    graph_path = "graph/aifs-graphcast.pt"
    feature_config_path = "hydraconfig/features/base.yaml"

    feature_config = fdl.Config(
        data.FeatureConfig.from_yaml,
        path=feature_config_path,
    )

    data_handler = fdl.Config(
        data.DataHandler,
        feature_config=feature_config,
        dataset_path=dataset_path,
    )

    graph_provider = fdl.Config(
        data.StaticGraphProvider,
        path=graph_path,
    )

    dataset = fdl.Config(
        data.AnemoiDataset,
        path=dataset_path,
        graph_provider=graph_provider,
    )

    dataloader = fdl.Config(
        DataLoader,
        dataset,
        batch_size=1,
        shuffle=True,
    )

    backbone = fdl.Config(
        Graphcast,
        feature_config,
    )

    optimizer_factory = fdl.Partial(
        Adam,
        lr=1e-4,  # Lower LR for larger model
    )

    #  scheduler_factory = fdl.Partial(
    #      StepLR,
    #      step_size=100,  # Drop LR less frequently
    #      gamma=0.5,  # Less aggressive LR reduction
    #  )

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
        #  scheduler_factory=scheduler_factory,
    )

    trainer = fdl.Config(
        Trainer,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
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

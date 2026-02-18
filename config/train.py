import fiddle as fdl
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader

from equicast import data, experiment
from equicast.callbacks import StepTimer, TimeDeltaCheckpoint
from equicast.data import EquivariantGraphDataHandler, FeatureConfig
from equicast.experiment import TrainConfig
from equicast.logger import MLFlowLogger
from equicast.model.backbones.painn import PaiNN
from equicast.model.model import EquivariantMSELoss, Model

LOCAL_DATASET_PATH = "/home/masc/storage/era5-o96-2024-tail200-6h.zarr"
GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"
EQUIVARIANT_FEATURE_CONFIG_PATH = "config/features/base_equivariant.yaml"



def default_logger():
    return fdl.Config(
        MLFlowLogger,
        experiment_name="masc1",
        tracking_uri="https://mlflow.dmidev.org/",
    )


def default_trainer(logger):
    return fdl.Config(
        Trainer,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        callbacks=[
            fdl.Config(
                TimeDeltaCheckpoint,
                save_initial=True,
            ),
            fdl.Config(StepTimer),
        ],
    )


def default_dataloader(dataset_path, graph_path):
    graph_provider = fdl.Config(
        data.StaticGraphProvider,
        graph_path=graph_path,
    )

    dataset = fdl.Config(
        data.AnemoiDataset,
        path=dataset_path,
        graph_provider=graph_provider,
    )

    return fdl.Config(
        DataLoader,
        dataset,
        num_workers=8,
        batch_size=1,
        shuffle=True,
    )


def default_backbone(data_handler):
    return fdl.Config(
        PaiNN,
        data_handler=data_handler,
        edges=[
            ("grid", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "mesh"),
            ("mesh", "to", "grid"),
        ],
        input_nodes="grid",
        hidden_dim=128,
    )


def main():
    loss_fn = fdl.Config(EquivariantMSELoss)
    feature_config = fdl.Config(
        FeatureConfig.from_yaml,
        path=EQUIVARIANT_FEATURE_CONFIG_PATH,
    )

    data_handler = fdl.Config(
        EquivariantGraphDataHandler,
        feature_config=feature_config,
        dataset_path=LOCAL_DATASET_PATH,
    )

    backbone = default_backbone(data_handler)

    model = fdl.Config(
        Model,
        backbone=backbone,
        loss_fn=loss_fn,
    )

    dataloader = default_dataloader(
        LOCAL_DATASET_PATH,
        GRAPH_PATH,
    )

    logger = default_logger()
    trainer = default_trainer(logger)
    cfg = fdl.Config(
        TrainConfig,
        model,
        trainer,
        dataloader,
        logger,
    )

    experiment.run_experiment(cfg)


if __name__ == "__main__":
    main()

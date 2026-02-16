import argparse
import importlib

import fiddle as fdl
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader

from equicast import data, experiments
from equicast.callbacks import StepTimer, TimeDeltaCheckpoint
from equicast.data import EquivariantGraphDataHandler, FeatureConfig
from equicast.experiments import TrainConfig
from equicast.logger import MLFlowLogger
from equicast.model.backbones.painn import PaiNN
from equicast.model.model import Model, equivariant_loss_fn
from equicast.utils.mlflow_loader import load_checkpoint_path_from_mlflow

LOCAL_DATASET_PATH = "/home/masc/storage/era5-o96-2024-tail200-6h.zarr"
GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"
EQUIVARIANT_FEATURE_CONFIG_PATH = "config/features/base_equivariant.yaml"


def import_fiddler(name):
    path = f"config.fiddlers.{name}"
    module = importlib.import_module(path)
    return module.fiddler


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
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--fiddler",
        action="append",
        default=[],
        help="Fiddlers, (repeatable)",
    )
    parser.add_argument(
        "--continue",
        type=str,
        default=None,
        dest="run_id",
        help="MLflow run ID to resume training from",
    )

    args, _ = parser.parse_known_args()

    loss_fn = equivariant_loss_fn
    feature_config = fdl.Config(FeatureConfig.from_yaml, path=EQUIVARIANT_FEATURE_CONFIG_PATH)
    data_handler = fdl.Config(
        EquivariantGraphDataHandler, feature_config=feature_config, dataset_path=LOCAL_DATASET_PATH
    )

    backbone = default_backbone(data_handler)

    model = fdl.Config(Model, backbone=backbone, loss_fn=loss_fn)
    dataloader = default_dataloader(LOCAL_DATASET_PATH, GRAPH_PATH)
    logger = default_logger()
    trainer = default_trainer(logger)
    cfg = fdl.Config(TrainConfig, model, trainer, dataloader, logger)

    for name in args.fiddler:
        fiddler = import_fiddler(name)
        fiddler(cfg)

    if args.run_id:
        cfg.ckpt_path = load_checkpoint_path_from_mlflow(args.run_id)
        cfg.logger.run_id = args.run_id
        cfg.trainer.logger.run_id = args.run_id

    experiments.run_experiment(cfg)


if __name__ == "__main__":
    main()

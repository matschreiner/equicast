import argparse
import importlib

import fiddle as fdl
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader

from equicast import data, experiments
from equicast.callbacks import TimeDeltaCheckpoint
from equicast.experiments import TrainConfig
from equicast.logger import MLFlowLogger
from equicast.model.model import Model

DATASET_PATH = "/home/masc/storage/mini_aifs.zarr"
GRAPH_PATH = "graph/aifs-graphcast.pt"

BACKBONES = {
    "painn": "config.backbones.painn",
    "equivariant_gnn": "config.backbones.equivariant_gnn",
    "simple_equivariant": "config.backbones.simple_equivariant",
    "gnn": "config.backbones.gnn",
    "encprocdec": "config.backbones.encprocdec",
    "graphcast": "config.backbones.graphcast",
}

FIDDLERS = {
    "leonardo": "config.fiddlers.leonardo",
    "local": "config.fiddlers.local",
    "debug": "config.fiddlers.debug",
    "cuda_profile": "config.fiddlers.cuda_profile",
}


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
        ],
    )


def default_dataloader(dataset_path, graph_path):
    graph_provider = fdl.Config(
        data.StaticGraphProvider,
        path=graph_path,
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


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--backbone",
        required=True,
        choices=BACKBONES.keys(),
        help="Backbone config to use",
    )
    parser.add_argument(
        "--fiddler",
        action="append",
        default=[],
        choices=FIDDLERS.keys(),
        help="Fiddlers to apply (repeatable)",
    )
    args, _ = parser.parse_known_args()

    backbone, loss_fn = importlib.import_module(BACKBONES[args.backbone]).backbone_config(
        DATASET_PATH
    )
    model = fdl.Config(Model, backbone=backbone, loss_fn=loss_fn)
    dataloader = default_dataloader(DATASET_PATH, GRAPH_PATH)
    logger = default_logger()
    trainer = default_trainer(logger)
    cfg = fdl.Config(TrainConfig, model, trainer, dataloader, logger)

    for name in args.fiddler:
        module = importlib.import_module(FIDDLERS[name])
        module.fiddler(cfg)

    experiments.run_experiment(cfg)


if __name__ == "__main__":
    main()

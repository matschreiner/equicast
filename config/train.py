import fiddle as fdl
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader

from equicast import TRACKING_URI, data, experiment
from equicast.callbacks import StepTimer, TimeDeltaCheckpoint
from equicast.data import EquivariantGraphDataHandler, FeatureConfig
from equicast.data.feature_index import FeatureIndex
from equicast.experiment import TrainConfig
from equicast.logger import MLFlowLogger
from equicast.model.backbones.painn import PaiNN
from equicast.model.model import EquivariantMSELoss, Model

DATASET_PATH = "storage/era5-o96-2024-tail200-6h.zarr"
GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"
FEATURE_CONFIG_PATH = "config/features/base_equivariant.yaml"


def default_logger():
    return fdl.Config(
        MLFlowLogger,
        experiment_name="equicast",
        tracking_uri=TRACKING_URI,
    )


def default_trainer(logger):
    return fdl.Config(
        Trainer,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        enable_checkpointing=False,  # don't use default checkpointing
        callbacks=[
            fdl.Config(TimeDeltaCheckpoint, save_initial=True),
            fdl.Config(StepTimer),
        ],
    )


def default_dataloader(dataset_path, graph_path):
    graph_provider = fdl.Config(data.StaticGraphProvider, graph_path=graph_path)
    dataset = fdl.Config(data.AnemoiDataset, path=dataset_path, graph_provider=graph_provider)
    return fdl.Config(DataLoader, dataset, num_workers=8, batch_size=1, shuffle=True)


def default_model(dataset_path, feature_config_path):
    feature_config = fdl.Config(FeatureConfig.from_yaml, path=feature_config_path)
    data_handler = fdl.Config(
        EquivariantGraphDataHandler,
        feature_config=feature_config,
        dataset_path=dataset_path,
    )
    feature_index = fdl.Config(
        FeatureIndex.from_dataset,
        dataset_path=dataset_path,
        feature_config=feature_config,
    )
    backbone = fdl.Config(
        PaiNN,
        feature_index=feature_index,
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
    return fdl.Config(Model, backbone=backbone, data_handler=data_handler, loss_fn=fdl.Config(EquivariantMSELoss))


def main():
    # NOTE: DATASET_PATH is passed to both default_model and default_dataloader.
    # Fiddlers that change the dataset must update both:
    #   cfg.model.backbone.feature_index.dataset_path = new_path
    #   cfg.model.data_handler.dataset_path = new_path
    #   cfg.dataloader.dataset.path = new_path
    logger = default_logger()
    cfg = fdl.Config(
        TrainConfig,
        model=default_model(DATASET_PATH, FEATURE_CONFIG_PATH),
        trainer=default_trainer(logger),
        dataloader=default_dataloader(DATASET_PATH, GRAPH_PATH),
        logger=logger,
    )

    experiment.run_experiment(cfg)


if __name__ == "__main__":
    main()

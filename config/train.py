import fiddle as fdl
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader

from config.backbones import build_painn
from config.fiddle_tags import DatasetPath, FeatureConfigPath, GraphPath
from equicast import TRACKING_URI, data, experiment
from equicast.callbacks import StepTimer, TimeDeltaCheckpoint
from equicast.data import EquivariantGraphDataHandler, FeatureConfig
from equicast.experiment import TrainConfig
from equicast.logger import MLFlowLogger
from equicast.metrics import WeatherBenchTracker
from equicast.model.losses import EquivariantMSELoss
from equicast.model.model import Model

DATASET_PATH = "storage/era5-o96-2024-tail200-6h.zarr"
GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"
FEATURE_CONFIG_PATH = "config/features/base_equivariant.yaml"


def base_config() -> fdl.Config[TrainConfig]:
    dataset_path = DatasetPath.new(default=DATASET_PATH)
    graph_path = GraphPath.new(default=GRAPH_PATH)
    feature_config_path = FeatureConfigPath.new(default=FEATURE_CONFIG_PATH)

    logger = fdl.Config(
        MLFlowLogger,
        experiment_name="equicast",
        tracking_uri=TRACKING_URI,
    )

    feature_config = fdl.Config(
        FeatureConfig.from_yaml,
        path=feature_config_path,
    )

    data_handler = fdl.Config(
        EquivariantGraphDataHandler,
        feature_config=feature_config,
        dataset_path=dataset_path,
    )

    backbone = fdl.Config(
        build_painn,
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

    weatherbench_tracker = fdl.Config(WeatherBenchTracker, data_handler=data_handler)

    model = fdl.Config(
        Model,
        backbone=backbone,
        data_handler=data_handler,
        loss_fn=fdl.Config(EquivariantMSELoss),
        metrics_tracker=weatherbench_tracker,
    )

    graph_provider = fdl.Config(
        data.StaticGraphProvider,
        graph_path=graph_path,
    )

    dataset = fdl.Config(
        data.AnemoiDataset,
        path=dataset_path,
        graph_provider=graph_provider,
    )

    dataloader = fdl.Config(
        DataLoader,
        dataset=dataset,
        num_workers=8,
        batch_size=1,
        shuffle=True,
    )

    trainer = fdl.Config(
        Trainer,
        logger=logger,
        log_every_n_steps=1,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        callbacks=[
            fdl.Config(TimeDeltaCheckpoint, save_initial=True),
            fdl.Config(StepTimer),
        ],
    )

    return fdl.Config(
        TrainConfig,
        model=model,
        trainer=trainer,
        dataloader=dataloader,
        logger=logger,
    )


def main():
    cfg = base_config()
    experiment.run_experiment(cfg)


if __name__ == "__main__":
    main()

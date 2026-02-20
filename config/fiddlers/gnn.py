import fiddle as fdl

from equicast import data
from equicast.data import FeatureConfig, FeatureIndex
from equicast.metrics import WeatherBenchTracker
from equicast.model.backbones.gnn import GNN
from equicast.model.model import MSELoss

FEATURE_CONFIG_PATH = "config/features/base.yaml"


def fiddler(cfg: fdl.Config) -> None:
    dataset_path = cfg.model.data_handler.dataset_path
    feature_config = fdl.Config(FeatureConfig.from_yaml, path=FEATURE_CONFIG_PATH)
    feature_index = fdl.Config(
        FeatureIndex.from_dataset,
        dataset_path=dataset_path,
        feature_config=feature_config,
    )

    cfg.model.backbone = fdl.Config(
        GNN,
        feature_index=feature_index,
        edges=cfg.model.backbone.edges,
        input_nodes="grid",
        hidden_dim=int(cfg.model.backbone.hidden_dim * 1.6),
    )
    cfg.model.data_handler = fdl.Config(
        data.GraphDataHandler,
        feature_config=feature_config,
        dataset_path=dataset_path,
    )
    cfg.model.loss_fn = fdl.Config(MSELoss)
    cfg.model.metrics_tracker = fdl.Config(WeatherBenchTracker, data_handler=cfg.model.data_handler)

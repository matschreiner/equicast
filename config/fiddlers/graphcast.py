import fiddle as fdl

from equicast import data
from equicast.data import FeatureConfig
from equicast.metrics import WeatherBenchTracker
from equicast.model.backbones.graphcast import GraphCast
from equicast.model.losses import MSELoss

FEATURE_CONFIG_PATH = "config/features/base.yaml"


def fiddler(cfg: fdl.Config) -> None:
    dataset_path = cfg.model.data_handler.dataset_path
    feature_config = fdl.Config(FeatureConfig.from_yaml, path=FEATURE_CONFIG_PATH)

    data_handler_cfg = fdl.Config(
        data.GraphDataHandler,
        feature_config=feature_config,
        dataset_path=dataset_path,
    )
    cfg.model.data_handler = data_handler_cfg
    cfg.model.backbone = fdl.Config(
        GraphCast,
        data_handler=data_handler_cfg,
        edges=cfg.model.backbone.edges,
        input_nodes="grid",
        hidden_dim=int(cfg.model.backbone.hidden_dim * 1.3),
    )
    cfg.model.loss_fn = fdl.Config(MSELoss)
    cfg.model.metrics_tracker = fdl.Config(WeatherBenchTracker, data_handler=data_handler_cfg)

import fiddle as fdl

from equicast.callbacks import StepTimer

GRAPH_PATH = "resources/graphs/stage_b/graph.pt"
DATASET_PATH = "/home/masc/storage/n320_mini.zarr"
FEATURE_CONFIG_PATH = "config/features/base_equivariant_n320.yaml"


def fiddler(cfg: fdl.Config) -> None:
    cfg.model.data_handler.dataset_path = DATASET_PATH
    cfg.model.data_handler.feature_config.path = FEATURE_CONFIG_PATH
    cfg.dataloader.dataset.path = DATASET_PATH
    cfg.dataloader.dataset.graph_provider.graph_path = GRAPH_PATH

    cfg.model.backbone.edges = [
        ("data", "to", "hidden"),
        ("hidden", "to", "hidden"),
        ("hidden", "to", "hidden"),
        ("hidden", "to", "hidden"),
        ("hidden", "to", "data"),
    ]
    cfg.model.backbone.input_nodes = "data"
    cfg.model.data_handler.nodes = "data"
    cfg.dataloader.dataset.node_name = "data"

    cfg.dataloader.num_workers = 0
    cfg.dataloader.shuffle = False

    cfg.trainer.precision = "bf16-mixed"
    cfg.trainer.max_steps = 2000
    cfg.trainer.callbacks = [fdl.Config(StepTimer)]
    cfg.logger.experiment_name = "benchmark_n320"

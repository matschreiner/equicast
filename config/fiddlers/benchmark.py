import fiddle as fdl

from config.fiddle_tags import DatasetPath, GraphPath
from equicast.callbacks import StepTimer

GRAPH_PATH = "resources/stage_a/graph.pt"
#  DATASET_PATH = "resources/benchmark.zarr"
#  GRAPH_PATH = "storage/benchmark/stage_a/graph.pt"
DATASET_PATH = "storage/benchmark/benchmark.zarr"


def fiddler(cfg: fdl.Config) -> None:
    fdl.set_tagged(cfg, tag=DatasetPath, value=DATASET_PATH)
    fdl.set_tagged(cfg, tag=GraphPath, value=GRAPH_PATH)

    cfg.dataloader.num_workers = 0
    cfg.dataloader.shuffle = False

    cfg.trainer.max_epochs = 1000
    cfg.trainer.callbacks = [fdl.Config(StepTimer)]
    cfg.logger.experiment_name = "benchmark"

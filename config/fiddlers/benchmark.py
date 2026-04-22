import fiddle as fdl

from equicast.callbacks import StepTimer

GRAPH_PATH = "resources/graphs/stage_a/graph.pt"
DATASET_PATH = "storage/benchmark/benchmark.zarr"


def fiddler(cfg: fdl.Config) -> None:
    cfg.trainer.precision = "bf16-mixed"

    cfg.model.data_handler.dataset_path = DATASET_PATH
    cfg.dataloader.dataset.path = DATASET_PATH
    cfg.dataloader.dataset.graph_provider.graph_path = GRAPH_PATH

    cfg.dataloader.num_workers = 0
    cfg.dataloader.shuffle = False

    cfg.trainer.max_steps = 10000
    cfg.trainer.callbacks = [fdl.Config(StepTimer)]
    cfg.logger.experiment_name = "benchmark"

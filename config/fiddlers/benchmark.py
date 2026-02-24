import fiddle as fdl

from equicast import data
from equicast.callbacks import StepTimer
from equicast.data.dataset import AnemoiDataset

GRAPH_PATH = "resources/graph.pt"
DATASET_PATH = "resources/benchmark.zarr"


def fiddler(cfg: fdl.Config) -> None:
    graph_provider = fdl.Config(data.StaticGraphProvider, graph_path=GRAPH_PATH)

    train_dataset = fdl.Config(
        AnemoiDataset,
        path=DATASET_PATH,
        graph_provider=graph_provider,
    )

    cfg.dataloader.dataset = train_dataset
    cfg.dataloader.num_workers = 0
    cfg.dataloader.shuffle = False

    cfg.model.data_handler.dataset_path = DATASET_PATH

    cfg.trainer.max_epochs = 1
    cfg.trainer.callbacks = [fdl.Config(StepTimer)]
    cfg.logger.experiment_name = "benchmark"

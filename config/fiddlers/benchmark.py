import fiddle as fdl
from torch_geometric.loader import DataLoader

from equicast import data
from equicast.callbacks import StepTimer
from equicast.data.dataset import BenchmarkDataset

GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"


def fiddler(cfg: fdl.Config) -> None:
    graph_provider = fdl.Config(data.StaticGraphProvider, graph_path=GRAPH_PATH)
    train_dataset = fdl.Config(
        BenchmarkDataset,
        samples_path="resources/samples",
        graph_provider=graph_provider,
    )

    #  val_dataset = fdl.Config(
    #      BenchmarkDataset,
    #      samples_path="resources/val_samples",
    #      graph_provider=graph_provider,
    #      length=1,
    #  )
    #  val_dataloader = fdl.Config(DataLoader, val_dataset, num_workers=0, batch_size=1, shuffle=False)
    #  cfg.val_dataloader = val_dataloader
    #  cfg.trainer.val_check_interval = 3

    cfg.dataloader.dataset = train_dataset
    cfg.dataloader.num_workers = 0
    cfg.dataloader.shuffle = False

    cfg.trainer.callbacks = [fdl.Config(StepTimer)]
    cfg.logger.experiment_name = "benchmark"

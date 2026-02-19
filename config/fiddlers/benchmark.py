import fiddle as fdl

from equicast import data
from equicast.data.dataset import BenchmarkDataset

GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"


def fiddler(cfg: fdl.Config) -> None:
    graph_provider = fdl.Config(data.StaticGraphProvider, graph_path=GRAPH_PATH)
    cfg.dataloader.dataset = fdl.Config(
        BenchmarkDataset,
        samples_path="resources/samples",
        graph_provider=graph_provider,
    )
    cfg.dataloader.num_workers = 0
    cfg.dataloader.shuffle = False
    cfg.logger.experiment_name = "benchmark"

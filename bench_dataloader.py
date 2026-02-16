"""Benchmark dataloader throughput."""

import time

from torch_geometric.loader import DataLoader

from equicast.data.dataset import AnemoiDataset
from equicast.data.graph_provider import StaticGraphProvider

DATASET_PATH = "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"
GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"

NUM_BATCHES = 200


def bench(num_workers):
    graph_provider = StaticGraphProvider(graph_path=GRAPH_PATH)
    dataset = AnemoiDataset(path=DATASET_PATH, graph_provider=graph_provider)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    it = iter(loader)
    # warm up
    next(it)

    start = time.time()
    for i, _ in enumerate(it):
        if i >= NUM_BATCHES - 1:
            break
    elapsed = time.time() - start

    print(f"num_workers={num_workers:2d}  {NUM_BATCHES} batches in {elapsed:.2f}s  ({NUM_BATCHES / elapsed:.1f} it/s)")


if __name__ == "__main__":
    for nw in [0, 1, 2, 4, 8]:
        bench(nw)

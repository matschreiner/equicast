"""Benchmark raw zarr read throughput with multiprocessing."""

import time
from multiprocessing import Process, Queue

import torch
from anemoi.datasets import open_dataset

DATASET_PATH = (
    "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"
)
NUM_READS = 200


def read_raw(data, idx):
    return data[idx : idx + 2]


def read_and_cast(data, idx):
    return torch.tensor(data[idx : idx + 2])


def read_cast_and_permute(data, idx):
    #  graph_path = "graph/aifs-graphcast-unnormed.pt"
    #  graph = torch.load(graph_path, weights_only=False)
    raw = data[idx : idx + 2]
    data = torch.from_numpy(raw).squeeze().permute(0, 2, 1)
    #  graph["grid"].data = tensor
    return data


def worker(path, indices, queue, read_fn):
    data = open_dataset(path)
    for idx in indices:
        _ = read_fn(data, idx)
    queue.put(len(indices))


def bench(num_workers, read_fn, label):
    indices = list(range(NUM_READS))
    chunks = [indices[i::num_workers] for i in range(num_workers)]
    queue = Queue()
    procs = [Process(target=worker, args=(DATASET_PATH, chunk, queue, read_fn)) for chunk in chunks]

    start = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    elapsed = time.time() - start

    print(f"{label:20s}  {NUM_READS} reads in {elapsed:.2f}s  ({NUM_READS / elapsed:.1f} reads/s)")


if __name__ == "__main__":
    bench(8, read_raw, "raw")
    bench(8, read_and_cast, "raw + torch.tensor")
    bench(8, read_cast_and_permute, "raw + torch.tensor + permute")

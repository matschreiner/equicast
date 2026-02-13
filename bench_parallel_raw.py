"""Benchmark raw zarr read throughput with multiprocessing."""

import time
from multiprocessing import Process, Queue

import torch
from anemoi.datasets import open_dataset

DATASET_PATH = "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"
NUM_READS = 200


def worker(path, indices, queue):
    data = open_dataset(path)
    for idx in indices:
        _ = torch.tensor(data[idx : idx + 2])
    queue.put(len(indices))


def bench(num_workers):
    indices = list(range(NUM_READS))

    if num_workers == 0:
        data = open_dataset(DATASET_PATH)
        _ = data[0:2]  # warm up
        start = time.time()
        for idx in indices:
            _ = data[idx : idx + 2]
        elapsed = time.time() - start
    else:
        chunks = [indices[i::num_workers] for i in range(num_workers)]
        queue = Queue()
        procs = [Process(target=worker, args=(DATASET_PATH, chunk, queue)) for chunk in chunks]

        start = time.time()
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        elapsed = time.time() - start

    print(f"workers={num_workers:2d}  {NUM_READS} reads in {elapsed:.2f}s  ({NUM_READS / elapsed:.1f} reads/s)")


if __name__ == "__main__":
    bench(8)

"""Benchmark raw zarr read + torch tensor cast."""

import time

import torch
from anemoi.datasets import open_dataset

DATASET_PATH = "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"
NUM_READS = 200


def main():
    data = open_dataset(DATASET_PATH)
    n = len(data)

    # warm up
    _ = torch.tensor(data[0:2])

    start = time.time()
    for i in range(NUM_READS):
        idx = i % (n - 1)
        _ = torch.tensor(data[idx : idx + 2])
    elapsed = time.time() - start

    print(f"{NUM_READS} reads in {elapsed:.2f}s  ({NUM_READS / elapsed:.1f} it/s)")


if __name__ == "__main__":
    main()

"""Save raw timeseries tensors from the Leonardo dataset to disk for local use."""

import numpy as np
from anemoi.datasets import open_dataset
from tqdm import tqdm

DATASET_PATH = (
    "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"
)
OUTPUT_PATH = "timeseries_raw.npz"
START_IDX = 0
STEPS = 56
SUBSAMPLE = 6


def main():
    data = open_dataset(DATASET_PATH)

    frames = []
    for i in tqdm(range(STEPS + 2), desc="Reading frames"):
        idx = (START_IDX + i) * SUBSAMPLE
        frames.append(data[idx].squeeze().T)

    # shape: [num_frames, nodes, features]
    raw = np.stack(frames)
    np.savez_compressed(OUTPUT_PATH, raw=raw)
    print(f"Saved {raw.shape} to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

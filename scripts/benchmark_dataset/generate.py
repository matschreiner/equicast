"""Save random (input, target) frame pairs from the dataset to disk."""

import os
import random

import torch
import zarr

DATASET_PATH = "storage/era5-o96-2024-tail200-6h.zarr"
STEP = 1


def save_samples(data, indices: list, out_path: str):
    os.makedirs(out_path, exist_ok=True)
    print(f"Saving {len(indices)} frames → {out_path}  (indices: {indices})")

    for i, idx in enumerate(indices):
        input_data = torch.from_numpy(data[idx, :, 0, :]).T       # [nodes, features]
        target_data = torch.from_numpy(data[idx + STEP, :, 0, :]).T

        path = f"{out_path}/sample_{i}.pt"
        torch.save({"input": input_data, "target": target_data}, path)
        print(f"  Saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    data = zarr.open(DATASET_PATH, "r")["data"]  # [time, features, 1, nodes]
    all_indices = random.sample(range(len(data) - STEP), 4)
    train_indices, val_indices = all_indices[:3], all_indices[3:]
    save_samples(data, indices=train_indices, out_path="resources/samples")
    save_samples(data, indices=val_indices, out_path="resources/val_samples")


if __name__ == "__main__":
    main()

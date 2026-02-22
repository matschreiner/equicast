"""Slice the last frame from the forecast zarr dataset and save as a new zarr dataset."""

import zarr

DATASET = "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-od-fc-oper-0001-mars-n320-2016-2024-1h-v1.zarr"
OUT = "one_frame.zarr"

src = zarr.open(DATASET, mode="r")
dst = zarr.open(OUT, mode="w")

zarr.copy_all(src, dst)
dst["data"][-1:] = src["data"][-1:]
dst["data"].resize(1, *src["data"].shape[1:])

print(f"Saved last frame to {OUT}")

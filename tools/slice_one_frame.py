"""Slice the last frame from the forecast zarr dataset and save as a new zarr dataset."""

import zarr

DATASET = "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-od-fc-oper-0001-mars-n320-2016-2024-1h-v1.zarr"
OUT = "one_frame.zarr"

src = zarr.open(DATASET, mode="r")
dst = zarr.open(OUT, mode="w")

dst.attrs.update(src.attrs)

for name, item in src.items():
    if name == "data":
        print(f"Slicing data[-1:] from shape {item.shape}...")
        dst.create_dataset("data", data=src["data"][-1:], chunks=(1, *src["data"].chunks[1:]))
        print(f"  done, saved shape {dst['data'].shape}")
    else:
        print(f"Copying {name} {item.shape}...")
        zarr.copy(item, dst, name=name)
        print(f"  done")

print(f"Saved to {OUT}")

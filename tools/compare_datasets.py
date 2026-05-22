"""Compare a few overlapping timestamps between the 6h and 1h O96 datasets.

Uses timestamps after 2000-01-01 to avoid the ERA5.1 correction period.
"""

import numpy as np
import zarr

DS6H = "storage/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
DS1H = "storage/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"

ds6 = zarr.open(DS6H, "r")
ds1 = zarr.open(DS1H, "r")

vars6 = ds6.attrs["variables"]
vars1 = ds1.attrs["variables"]
common_vars = [v for v in vars6 if v in vars1]
idx6 = [vars6.index(v) for v in common_vars]
idx1 = [vars1.index(v) for v in common_vars]
print(f"Common variables: {len(common_vars)}")

dates6 = ds6["dates"][:]
dates1 = ds1["dates"][:]

cutoff = np.datetime64("2000-01-01")
common_dates = np.intersect1d(dates6[dates6 >= cutoff], dates1[dates1 >= cutoff])
print(f"Common timestamps after 2000: {len(common_dates)}")

if len(common_dates) == 0:
    print("No overlap.")
    exit()

rng = np.random.default_rng(42)
samples = np.sort(rng.choice(common_dates, size=min(5, len(common_dates)), replace=False))

nodes = np.linspace(0, ds6["data"].shape[-1] - 1, 500, dtype=int)

for date in samples:
    i6 = np.searchsorted(dates6, date)
    i1 = np.searchsorted(dates1, date)
    a = ds6["data"][i6][idx6][:, 0, :][:, nodes]
    b = ds1["data"][i1][idx1][:, 0, :][:, nodes]
    diff = np.abs(a - b).mean()
    print(f"  {date}  mean_abs_diff={diff:.6f}")

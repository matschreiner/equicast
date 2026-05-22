"""Compare a few overlapping timestamps between the 6h and 1h O96 datasets.

Uses timestamps after 2000-01-01 to avoid the ERA5.1 correction period.
"""

import numpy as np
import zarr

DS6H = "storage/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr"
DS1H = "storage/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"
N_SAMPLES = 5

ds6 = zarr.open(DS6H, "r")
ds1 = zarr.open(DS1H, "r")

dates6 = ds6["dates"][:].astype(str)
dates1 = ds1["dates"][:].astype(str)

common = sorted(set(dates6) & set(dates1) & {d for d in dates6 if d >= "2000-01-01"})
print(f"Common timestamps after 2000: {len(common)}")

if not common:
    print("No overlap.")
    exit()

rng = np.random.default_rng(42)
samples = rng.choice(common, size=min(N_SAMPLES, len(common)), replace=False)

dates6_list = list(dates6)
dates1_list = list(dates1)

for date in sorted(samples):
    i6 = dates6_list.index(date)
    i1 = dates1_list.index(date)
    a = ds6["data"][i6, :, 0, :]
    b = ds1["data"][i1, :, 0, :]
    n = min(a.shape[0], b.shape[0])
    diff = np.abs(a[:n] - b[:n]).mean()
    print(f"  {date}  mean_abs_diff={diff:.6f}")

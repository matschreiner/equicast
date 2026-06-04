"""Compute day-of-year climatology from an anemoi-format zarr dataset.

For each (day-of-year, hour) slot, computes the mean over all years in the
reference period. Output has 1464 slots (366 days × 4 six-hourly steps).

Slot index: (day_of_year - 1) * 4 + hour // 6   (day_of_year is 1-366)

Usage:
    python tools/compute_climatology.py \\
        storage/aifs-ea-an-oper-0001-mars-o96-1979-2023-6h-v8.zarr \\
        --output storage/climatology_o96_1990_2019.npz \\
        --start-year 1990 --end-year 2019

Output (npz):
    clim       — (1464, n_vars, n_nodes) float32, mean per slot
    variables  — (n_vars,) str
    latitudes  — (n_nodes,) float32
    longitudes — (n_nodes,) float32
"""

import argparse
import os
import sys

import numpy as np
from anemoi.datasets import open_dataset

N_SLOTS = 366 * 4  # 1464


def date_to_slot(date) -> int:
    """Convert a numpy datetime64 to slot index 0..1463."""
    ts = date.astype("datetime64[ms]").astype(object)
    doy = int(ts.strftime("%j"))   # 1-366
    hour = ts.hour                 # 0, 6, 12, 18
    return (doy - 1) * 4 + hour // 6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to anemoi zarr dataset")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year",   type=int, default=2019)
    parser.add_argument("--vars", nargs="+", default=None,
                        help="Variables to include (default: all non-cossin)")
    args = parser.parse_args()

    print(f"Opening {args.dataset}")
    ds = open_dataset(args.dataset)

    all_vars = list(ds.variables)
    _skip = {"cos_julian_day", "sin_julian_day", "cos_local_time", "sin_local_time",
             "cos_latitude", "sin_latitude", "cos_longitude", "sin_longitude",
             "insolation", "lsm", "sdor", "slor"}

    if args.vars:
        var_list = args.vars
    else:
        var_list = [v for v in all_vars if v not in _skip]

    var_indices = [ds.name_to_index[v] for v in var_list]
    n_vars  = len(var_list)
    n_nodes = ds.latitudes.shape[0]
    print(f"Variables ({n_vars}): {var_list}")
    print(f"Nodes: {n_nodes}")

    # Two-pass accumulation: sum and count per slot
    acc   = np.zeros((N_SLOTS, n_vars, n_nodes), dtype=np.float64)
    count = np.zeros(N_SLOTS, dtype=np.int32)

    n_total = len(ds.dates)
    n_used  = 0
    print(f"Scanning {n_total} timesteps for years {args.start_year}–{args.end_year}...")

    for i, date in enumerate(ds.dates):
        ts = date.astype("datetime64[ms]").astype(object)
        year = ts.year
        if year < args.start_year or year > args.end_year:
            continue

        slot = date_to_slot(date)
        # ds[i] shape: (n_all_vars, 1, n_nodes) — squeeze ensemble dim
        frame = ds[i].squeeze(axis=1)          # (n_all_vars, n_nodes)
        acc[slot]   += frame[var_indices]       # (n_vars, n_nodes)
        count[slot] += 1
        n_used += 1

        if n_used % 1000 == 0:
            print(f"  {n_used} timesteps processed...")

    print(f"Done. {n_used} timesteps used.")

    zero_slots = np.where(count == 0)[0]
    if len(zero_slots):
        print(f"Warning: {len(zero_slots)} slots with no data (e.g. leap day in non-leap datasets)")

    clim = np.where(count[:, None, None] > 0,
                    acc / np.maximum(count[:, None, None], 1),
                    0.0).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        clim=clim,
        variables=np.array(var_list),
        latitudes=ds.latitudes.astype(np.float32),
        longitudes=ds.longitudes.astype(np.float32),
        slot_count=count,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    print(f"Saved to {args.output}  (shape: {clim.shape})")


if __name__ == "__main__":
    main()

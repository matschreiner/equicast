#!/usr/bin/env python3
"""Slice and subsample an anemoi dataset (zarr store) along the time axis.

Usage:
    python tools/slice_dataset.py INPUT OUTPUT --slice "-3000:" --subsample 6
"""

import argparse
import sys

import numpy as np
import zarr

# python scripts/slice_anemoi_dataset.py /leonardo_work/DestE_340_26/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr test_dataset.zarr --slice="-60:" --subsample 6


def parse_slice(s: str) -> slice:
    """Parse a Python slice string like '-3000:', '100:2000', ':' into a slice object."""
    parts = s.split(":")
    if len(parts) == 1:
        # Single integer → single element slice
        i = int(parts[0])
        return slice(i, i + 1 if i != -1 else None)
    if len(parts) == 2:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        return slice(start, stop)
    if len(parts) == 3:
        start = int(parts[0]) if parts[0] else None
        stop = int(parts[1]) if parts[1] else None
        step = int(parts[2]) if parts[2] else None
        return slice(start, stop, step)
    raise ValueError(f"Invalid slice string: {s!r}")


def main():
    parser = argparse.ArgumentParser(description="Slice and subsample an anemoi zarr dataset.")
    parser.add_argument("input", help="Path to source .zarr store")
    parser.add_argument("output", help="Path to destination .zarr store")
    parser.add_argument(
        "--slice", default=":", dest="slice_str", help='Python slice notation, e.g. "-3000:" (default: ":")'
    )
    parser.add_argument("--subsample", type=int, default=1, help="Subsampling step (default: 1)")
    args = parser.parse_args()

    if args.subsample < 1:
        print("Error: --subsample must be >= 1", file=sys.stderr)
        sys.exit(1)

    print(f"Opening source: {args.input}")
    src = zarr.open(args.input, mode="r")
    print(f"Opening destination: {args.output}")
    dst = zarr.open(args.output, mode="w")

    total_len = src.data.shape[0]
    sl = parse_slice(args.slice_str)
    indices = np.arange(total_len)[sl][:: args.subsample]

    if len(indices) == 0:
        print("Error: slice/subsample results in 0 frames", file=sys.stderr)
        sys.exit(1)

    print(f"Source: {total_len} frames, shape {src.data.shape}")
    print(f"Slice: {args.slice_str}, subsample: {args.subsample}")
    print(f"Output: {len(indices)} frames (indices {indices[0]}..{indices[-1]})")

    # Copy temporal arrays in chunks to limit memory usage
    chunk_size = 500
    n_out = len(indices)

    # Create output data array
    out_shape = (n_out,) + src.data.shape[1:]
    dst.create_dataset("data", shape=out_shape, dtype=src.data.dtype, chunks=(1,) + src.data.shape[1:])

    print(f"Copying data ({n_out} frames, chunk_size={chunk_size})...")
    for start in range(0, n_out, chunk_size):
        end = min(start + chunk_size, n_out)
        idx_chunk = indices[start:end]
        dst.data[start:end] = src.data.get_orthogonal_selection(idx_chunk)
        print(f"  [{end}/{n_out}] frames copied", flush=True)
    print("  Data copy complete.")

    # Copy dates (use orthogonal selection to avoid loading all 30+ years)
    print("Copying dates...")
    out_dates = src.dates.get_orthogonal_selection(indices)
    dst.create_dataset("dates", data=out_dates, chunks=(n_out,))
    print(f"  Date range: {out_dates[0]} -> {out_dates[-1]}")

    # Copy all non-temporal arrays/groups verbatim
    temporal = {"data", "dates"}
    print(f"Source contains: {list(src.keys())}")
    for name in src:
        if name not in temporal:
            print(f"  Copying '{name}' ...", end=" ", flush=True)
            dst.create_dataset(name, data=src[name][:], compressor=None)
            print(f"shape={src[name].shape}")

    # Copy attributes, updating temporal metadata
    print(f"Copying {len(src.attrs)} attributes...")
    for k, v in src.attrs.items():
        dst.attrs[k] = v
        print(f"  {k}: {str(v)[:80]}")

    # Update frequency
    if "frequency" in src.attrs and args.subsample > 1:
        from anemoi.utils.dates import frequency_to_string, frequency_to_timedelta

        orig_freq = frequency_to_timedelta(src.attrs["frequency"])
        new_freq = orig_freq * args.subsample
        dst.attrs["frequency"] = frequency_to_string(new_freq)
        print(f"Updated frequency: {src.attrs['frequency']} -> {dst.attrs['frequency']}")

    # Update date bounds
    first = str(np.datetime_as_string(out_dates[0], unit="s"))
    last = str(np.datetime_as_string(out_dates[-1], unit="s"))
    dst.attrs["first_date"] = first
    dst.attrs["last_date"] = last
    print(f"Updated date bounds: {first} -> {last}")

    print("Done.")


if __name__ == "__main__":
    main()

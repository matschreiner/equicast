"""Create a spillover-test dataset: first n_input frames intact, subsequent frames
have all non-forcing variables set to zero.

Usage:
    python tools/make_spillover_dataset.py storage/o96-eval-2023.zarr storage/o96-spillover.zarr
"""

import argparse

import numpy as np
import zarr
from anemoi.datasets import open_dataset
from tqdm import tqdm

FORCING_VARS = {
    "cos_latitude", "cos_longitude", "sin_latitude", "sin_longitude",
    "cos_julian_day", "cos_local_time", "lsm", "sin_julian_day",
    "sin_local_time", "sdor", "slor",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Source zarr dataset")
    parser.add_argument("output", help="Output zarr dataset")
    parser.add_argument("--n-input", type=int, default=2, help="Number of intact input frames (default: 2)")
    parser.add_argument("--n-frames", type=int, default=20, help="Total frames per window (default: 20)")
    args = parser.parse_args()

    src = zarr.open(args.input, mode="r")
    dst = zarr.open(args.output, mode="w")

    ds = open_dataset(args.input)
    variable_names = list(ds.variables)
    forcing_idxs = {i for i, name in enumerate(variable_names) if name in FORCING_VARS}
    non_forcing_idxs = [i for i in range(len(variable_names)) if i not in forcing_idxs]

    print(f"Variables: {len(variable_names)} total, {len(forcing_idxs)} forcing, {len(non_forcing_idxs)} non-forcing")
    print(f"Forcing: {[variable_names[i] for i in sorted(forcing_idxs)]}")

    n_total = src.data.shape[0]
    n_windows = n_total - args.n_frames + 1
    out_shape = (n_total,) + src.data.shape[1:]
    dst.create_dataset("data", shape=out_shape, dtype=src.data.dtype, chunks=(1,) + src.data.shape[1:])

    print(f"Copying {n_total} frames, zeroing non-forcing in frames {args.n_input}+...")
    for i in tqdm(range(n_total)):
        frame = src.data[i]  # (n_features, 1, n_nodes)
        if i >= args.n_input:
            frame = frame.copy()
            frame[non_forcing_idxs] = 0.0
        dst.data[i] = frame

    # Copy all non-data arrays verbatim
    for name in src:
        if name != "data":
            item = src[name]
            if isinstance(item, zarr.hierarchy.Group):
                zarr.copy(item, dst, name=name)
            else:
                dst.create_dataset(name, data=item[:], compressor=None)

    for k, v in src.attrs.items():
        dst.attrs[k] = v

    print("Done.")


if __name__ == "__main__":
    main()

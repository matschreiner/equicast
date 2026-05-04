"""Visualize a variable from an anemoi zarr dataset.

Usage:
    python tools/visualize_dataset.py resources/benchmark.zarr z_500 --frames 0:5 --output /tmp/out.mp4
    python tools/visualize_dataset.py resources/benchmark.zarr 2t --frames 0:1 --output /tmp/out.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from anemoi.datasets import open_dataset

from equicast.visualization import make_video, plot_field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Path to anemoi zarr dataset")
    parser.add_argument("feature", help="Variable name (e.g. z_500, 2t)")
    parser.add_argument("--frames", default=None, help="Frame range as START:END (default: all)")
    parser.add_argument("--output", default=None, help="Output path (.png or .mp4)")
    args = parser.parse_args()

    ds = open_dataset(args.dataset_path)

    if args.feature not in ds.name_to_index:
        available = sorted(ds.name_to_index.keys())
        raise ValueError(f"'{args.feature}' not found. Available: {available}")

    feat_idx = ds.name_to_index[args.feature]
    latlon = np.stack([ds.latitudes, ds.longitudes], axis=-1)

    start, end = (map(int, args.frames.split(":")) if args.frames else (0, len(ds)))
    frames = np.stack([ds[t][:, 0, :][feat_idx] for t in range(start, end)])

    if len(frames) == 1:
        ax, _ = plot_field(frames[0], latlon, title=f"{args.feature} — frame {start}")
        if args.output:
            ax.figure.savefig(args.output, bbox_inches="tight")
            print(f"Saved to {args.output}")
        else:
            plt.show()
    else:
        output = args.output or f"{args.feature}_{start}_{end}.mp4"
        make_video(frames, latlon, output_path=output, title_template=f"{args.feature} — frame {{frame}}")
        print(f"Saved to {output}")


if __name__ == "__main__":
    main()

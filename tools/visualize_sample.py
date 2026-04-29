"""Visualize a single sample as a static map.

Usage:
    python scripts/visualize_sample.py samples/26.04.23_13:29:19 z_500
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import torch
import yaml

from equicast.visualization import plot_field

FEATURE_CONFIG = "config/features/base_equivariant.yaml"
GRAPH_PATH = "resources/graphs/stage_a/graph.pt"
NODES = "grid"


def get_feature_index(feature_name: str) -> int:
    with open(FEATURE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    features = cfg["prognostic"]
    if feature_name not in features:
        raise ValueError(f"'{feature_name}' not found. Available: {features}")
    return features.index(feature_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_dir", help="Path to sample directory")
    parser.add_argument("feature", help="Feature name to visualize (e.g. z_500)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    feature_idx = get_feature_index(args.feature)

    sample = torch.load(sample_dir / "sample.pt", weights_only=True)[:, feature_idx].numpy()
    truth_path = sample_dir / "ground_truth.pt"
    ground_truth = torch.load(truth_path, weights_only=True)[:, feature_idx].numpy() if truth_path.exists() else None

    graph = torch.load(GRAPH_PATH, weights_only=False)
    latlon = graph[NODES].x.numpy()

    vmin = min(sample.min(), ground_truth.min() if ground_truth is not None else sample.min())
    vmax = max(sample.max(), ground_truth.max() if ground_truth is not None else sample.max())

    n_panels = 2 if ground_truth is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 6 * n_panels),
                             subplot_kw=dict(projection=ccrs.PlateCarree()))
    if n_panels == 1:
        axes = [axes]

    _, im = plot_field(sample, latlon, ax=axes[0], title=f"{args.feature} – muestra", vmin=vmin, vmax=vmax)
    if ground_truth is not None:
        plot_field(ground_truth, latlon, ax=axes[1], title=f"{args.feature} – ground truth", vmin=vmin, vmax=vmax)

    plt.colorbar(im, ax=axes, orientation="horizontal", pad=0.05, aspect=40)

    output_path = args.output or str(sample_dir / f"{args.feature}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Imagen guardada → {output_path}")


if __name__ == "__main__":
    main()

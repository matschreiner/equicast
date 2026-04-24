"""Create a video from a diffusion sampling trajectory.

Usage:
    python scripts/visualize_trajectory.py samples/26.04.23_13:29:19 z_500
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from equicast.visualization import make_video


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
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--normalize-frames", action="store_true", help="Normalize each frame independently")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir)
    trajectory = torch.load(sample_dir / "trajectory.pt", weights_only=True)  # [steps, nodes, features]

    feature_idx = get_feature_index(args.feature)
    fields = trajectory[:, :, feature_idx].numpy()  # [steps, nodes]

    if args.normalize_frames:
        vmin = vmax = None
    else:
        vmin, vmax = fields.min(), fields.max()

    graph = torch.load(GRAPH_PATH, weights_only=False)
    latlon = graph[NODES].x.numpy()

    output_path = args.output or str(sample_dir / f"{args.feature}.mp4")
    make_video(
        fields=fields,
        latlon=latlon,
        output_path=output_path,
        fps=args.fps,
        title_template=f"{args.feature} – paso {{frame}}",
        vmin=vmin,
        vmax=vmax,
    )
    print(f"Vídeo guardado → {output_path}")


if __name__ == "__main__":
    main()

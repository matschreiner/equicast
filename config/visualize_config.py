"""Visualize scalar fields with vector overlays."""

import fiddle as fdl
import matplotlib.pyplot as plt

from equicast import data
from equicast.experiments.config import vis_config
from equicast.visualization import (
    make_video_with_vectors,
    plot_field_with_vectors,
)


def get_graph(dataset, idx: int = 0):
    """Get a single input graph from the dataset."""
    sample = dataset[idx]
    return sample["input"]


def get_graphs(dataset, data_handler, start_idx: int = 0, num_frames: int = 20):
    """Get multiple prepared graphs from the dataset."""
    graphs = []
    for i in range(start_idx, start_idx + num_frames):
        sample = dataset[i]
        graph = data_handler.prepare_input(sample["input"])
        graphs.append(graph)
    return graphs


def visualize(
    graph,
    data_handler,
    vector_field: str,
    scalar_field: str,
    output_path: str = None,
    subsample: int = 20,
    scale: float = None,
    title: str = None,
    cmap: str = "coolwarm",
    vmin: float = None,
    vmax: float = None,
):
    """Run visualization with the given configuration."""
    # Prepare graph with vector features
    graph = data_handler.prepare_input(graph)

    ax, im, quiv = plot_field_with_vectors(
        graph=graph,
        data_handler=data_handler,
        vector_field=vector_field,
        scalar_field=scalar_field,
        subsample=subsample,
        scale=scale,
        title=title,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=40)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    return ax, im, quiv


def create_video(
    graphs,
    data_handler,
    vector_field: str,
    scalar_field: str,
    output_path: str,
    fps: int = 5,
    subsample: int = 20,
    arrow_length: float = 1.0,
    title_template=None,
    cmap: str = "coolwarm",
    center=None,
    extent=None,
    normalize_vectors: bool = False,
):
    """Create video with vectors."""
    if title_template is None:
        title_template = (
            lambda i: f"{scalar_field} with {vector_field} - Frame {i}"
        )

    anim = make_video_with_vectors(
        graphs=graphs,
        data_handler=data_handler,
        vector_field=vector_field,
        scalar_field=scalar_field,
        output_path=output_path,
        fps=fps,
        subsample=subsample,
        arrow_length=arrow_length,
        title_template=title_template,
        cmap=cmap,
        center=center,
        extent=extent,
        normalize_vectors=normalize_vectors,
    )
    print(f"Saved video to {output_path}")
    return anim


def main():
    dataset_path = "/home/masc/storage/mini_aifs.zarr"
    graph_path = "graph/graphcast-paper.pt"
    feature_config_path = "hydraconfig/features/base_equivariant.yaml"

    feature_config = fdl.Config(
        data.FeatureConfig.from_yaml,
        path=feature_config_path,
    )

    data_handler = fdl.Config(
        data.EquivariantGraphDataHandler,
        feature_config=feature_config,
        dataset_path=dataset_path,
    )

    graph_provider = fdl.Config(
        data.StaticGraphProvider,
        path=graph_path,
    )

    dataset = fdl.Config(
        data.AnemoiDataset,
        path=dataset_path,
        graph_provider=graph_provider,
    )

    graphs = fdl.Config(
        get_graphs,
        dataset=dataset,
        data_handler=data_handler,
        start_idx=0,
        num_frames=200,
    )

    video_cfg = fdl.Config(
        create_video,
        graphs=graphs,
        data_handler=data_handler,
        vector_field="wind_1000",
        scalar_field="z_1000",
        output_path="video/equivariant.mp4",
        fps=5,
        subsample=1,
        arrow_length=1.0,  # Higher = longer arrows
        cmap="coolwarm",
        # Zoom: center=(lat, lon), extent=(width, height) in degrees
        # Set to None for global view
        center=(55, 10),  # Europe
        extent=(60, 40),
        normalize_vectors=True,  # Show direction only (same arrow length)
    )

    # Visualize config graph
    vis_config(video_cfg)

    # Build and run
    fdl.build(video_cfg)


if __name__ == "__main__":
    main()

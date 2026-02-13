"""Save a timeseries from the Leonardo dataset to disk for local use."""

import torch
from anemoi.datasets import open_dataset

DATASET_PATH = (
    "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"
)
GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"
OUTPUT_PATH = "timeseries.pt"
START_IDX = 0
STEPS = 56
SUBSAMPLE = 6


def main():
    data = open_dataset(DATASET_PATH)
    graph = torch.load(GRAPH_PATH, weights_only=False)

    timeseries = []
    for i in range(STEPS + 1):
        input_idx = (START_IDX + i) * SUBSAMPLE
        target_idx = input_idx + SUBSAMPLE

        input_raw = data[input_idx].squeeze().T
        target_raw = data[target_idx].squeeze().T

        input_graph = graph.clone()
        target_graph = graph.clone()
        input_graph["grid"].data = torch.from_numpy(input_raw)
        target_graph["grid"].data = torch.from_numpy(target_raw)

        timeseries.append({"input": input_graph, "target": target_graph})

    torch.save(timeseries, OUTPUT_PATH)
    print(f"Saved {len(timeseries)} frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

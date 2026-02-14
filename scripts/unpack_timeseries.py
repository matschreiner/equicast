"""Unpack raw timeseries npz into graph timeseries and save as .pt."""

import numpy as np
import torch

RAW_PATH = "timeseries_raw.npz"
GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"
OUTPUT_PATH = "timeseries.pt"


def main():
    raw = np.load(RAW_PATH)["raw"]
    graph = torch.load(GRAPH_PATH, weights_only=False)

    timeseries = []
    for i in range(len(raw) - 1):
        input_graph = graph.clone()
        target_graph = graph.clone()
        input_graph["grid"].data = torch.from_numpy(raw[i]).float()
        target_graph["grid"].data = torch.from_numpy(raw[i + 1]).float()
        timeseries.append({"input": input_graph, "target": target_graph})

    torch.save(timeseries, OUTPUT_PATH)
    print(f"Saved {len(timeseries)} frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

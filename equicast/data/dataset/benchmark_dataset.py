import glob

import torch
from torch.utils.data import Dataset


class BenchmarkDataset(Dataset):
    """Loads pre-saved frame pairs into memory and returns them indefinitely.

    Samples are plain dicts {"input": Tensor, "target": Tensor} produced by
    scripts/sample_dataset.py. The graph structure is constructed once at init
    using the provided graph_provider.
    """

    def __init__(self, samples_path: str, graph_provider, length: int = 10000):
        paths = sorted(glob.glob(f"{samples_path}/sample_*.pt"))
        if not paths:
            raise FileNotFoundError(f"No sample_*.pt files found in {samples_path}")

        self.samples = []
        for path in paths:
            tensors = torch.load(path, weights_only=True)
            graph = graph_provider.get_graph(0)
            input_graph = graph.clone()
            target_graph = graph.clone()
            input_graph["grid"].data = tensors["input"]
            target_graph["grid"].data = tensors["target"]
            self.samples.append({"input": input_graph, "target": target_graph})

        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.samples[idx % len(self.samples)]

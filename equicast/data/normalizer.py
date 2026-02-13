import torch


class Normalizer(torch.nn.Module):
    """Z-score normalizer for selected input/output features."""

    def __init__(self, statistics: dict[str, torch.Tensor], in_idxs: list[int], out_idxs: list[int]):
        super().__init__()
        mean = statistics["mean"]
        std = statistics["stdev"]

        self.register_buffer("in_mean", mean[in_idxs])
        self.register_buffer("in_std", std[in_idxs])
        self.register_buffer("out_mean", mean[out_idxs])
        self.register_buffer("out_std", std[out_idxs])

    def normalize_input(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.in_mean) / self.in_std  # type: ignore

    def normalize_output(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.out_mean) / self.out_std  # type: ignore

    def denormalize_output(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.out_std + self.out_mean  # type: ignore


class VectorNormalizer(torch.nn.Module):
    """Normalizes vector features by their mean norm."""

    def __init__(self, vector_mean_norm: torch.Tensor):
        super().__init__()
        self.register_buffer("vector_mean_norm", vector_mean_norm)

    def normalize_vectors(self, data: torch.Tensor) -> torch.Tensor:
        return data / self.vector_mean_norm.view(1, -1, 1)

    def denormalize_vectors(self, data: torch.Tensor) -> torch.Tensor:
        return data * self.vector_mean_norm.view(1, -1, 1)

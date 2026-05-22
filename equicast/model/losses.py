import torch


class MSELoss(torch.nn.Module):
    def forward(self, backbone_output, backbone_target):
        return torch.nn.functional.mse_loss(backbone_output, backbone_target)


class WeightedMSELoss(torch.nn.Module):
    """MSE loss with per-variable weights.

    Usage in config: _target_: equicast.model.losses.WeightedMSELoss
                     variable_weights: {z: 12.0, t: 6.0, sp: 10.0}
    Prefix matching: "t" weights t_850, t_500, etc.
    Variables not listed default to weight 1.0.
    Call loss_fn.build(data_handler) after instantiation to finalise weights.
    """

    def __init__(self, variable_weights: dict[str, float]):
        super().__init__()
        self.variable_weights = variable_weights

    def build(self, data_handler) -> "WeightedMSELoss":
        weights = _build_scalar_weights(
            self.variable_weights, data_handler.out_idxs, data_handler.name_to_index
        )
        self.register_buffer("weights", weights)
        return self

    def forward(self, backbone_output, backbone_target):
        # backbone_output, backbone_target: [nodes, out_dim]
        sq_err = (backbone_output - backbone_target) ** 2
        return (sq_err * self.weights).mean()


def _build_scalar_weights(variable_weights: dict[str, float], out_idxs: list[int], name_to_index: dict[str, int]) -> torch.Tensor:
    index_to_name = {v: k for k, v in name_to_index.items()}
    weights = torch.ones(len(out_idxs))
    for i, raw_idx in enumerate(out_idxs):
        name = index_to_name[raw_idx]
        for pattern, w in variable_weights.items():
            if name == pattern or name.startswith(pattern + "_"):
                weights[i] = w
                break
    return weights


def _build_vector_weights(variable_weights: dict[str, float], vector_names: list[str]) -> torch.Tensor:
    weights = torch.ones(len(vector_names))
    for i, name in enumerate(vector_names):
        for pattern, w in variable_weights.items():
            if name == pattern or name.startswith(pattern + "_"):
                weights[i] = w
                break
    return weights


class EquivariantMSELoss(torch.nn.Module):
    def forward(self, backbone_output, backbone_target):
        scalar_loss = torch.nn.functional.mse_loss(backbone_output["scalar"], backbone_target["scalar"])
        vector_loss = torch.nn.functional.mse_loss(backbone_output["vector"], backbone_target["vector"])
        return scalar_loss + vector_loss


class WeightedEquivariantMSELoss(torch.nn.Module):
    """Equivariant MSE loss with per-variable weights.

    Usage in config: _target_: equicast.model.losses.WeightedEquivariantMSELoss
                     variable_weights: {z: 12.0, t: 6.0, wind: 0.6, sp: 10.0}
    Prefix matching: "wind" weights wind_850, wind_500, wind_10, etc.
    Variables not listed default to weight 1.0.
    Call loss_fn.build(data_handler) after instantiation to finalise weights.
    """

    def __init__(self, variable_weights: dict[str, float]):
        super().__init__()
        self.variable_weights = variable_weights

    def build(self, data_handler) -> "WeightedEquivariantMSELoss":
        scalar_w = _build_scalar_weights(
            self.variable_weights, data_handler.out_idxs, data_handler.name_to_index
        )
        self.register_buffer("scalar_weights", scalar_w)

        feature_config = data_handler.feature_config
        vector_names = list(feature_config.prognostic_vector.keys()) + list(
            getattr(feature_config, "diagnostic_vector", {}).keys()
        )
        vector_w = _build_vector_weights(self.variable_weights, vector_names)
        # [n_vectors] -> [1, n_vectors, 1] to broadcast over [nodes, n_vectors, 2]
        self.register_buffer("vector_weights", vector_w.view(1, -1, 1))
        return self

    def forward(self, backbone_output, backbone_target):
        scalar_sq_err = (backbone_output["scalar"] - backbone_target["scalar"]) ** 2
        vector_sq_err = (backbone_output["vector"] - backbone_target["vector"]) ** 2
        scalar_loss = (scalar_sq_err * self.scalar_weights).mean()
        vector_loss = (vector_sq_err * self.vector_weights).mean()
        return scalar_loss + vector_loss

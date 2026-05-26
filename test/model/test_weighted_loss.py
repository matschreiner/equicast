import pytest
import torch
from types import SimpleNamespace

from equicast.model.losses import WeightedMSELoss, WeightedEquivariantMSELoss


def make_scalar_handler(variables: list[str], out_names: list[str]):
    name_to_index = {name: i for i, name in enumerate(variables)}
    out_idxs = [name_to_index[n] for n in out_names]
    return SimpleNamespace(name_to_index=name_to_index, out_idxs=out_idxs)


def make_equivariant_handler(variables: list[str], out_names: list[str], prognostic_vector: dict):
    handler = make_scalar_handler(variables, out_names)
    handler.feature_config = SimpleNamespace(
        prognostic_vector=prognostic_vector,
        diagnostic_vector={},
    )
    return handler


# --- WeightedMSELoss ---

class TestWeightedMSELoss:
    def test_weight_tensor_shape(self):
        handler = make_scalar_handler(["z_500", "t_850", "sp"], ["z_500", "t_850", "sp"])
        loss = WeightedMSELoss({"z": 12.0, "t": 6.0}).build(handler)
        assert loss.weights.shape == (3,)

    def test_exact_name_match(self):
        handler = make_scalar_handler(["sp", "t_850"], ["sp", "t_850"])
        loss = WeightedMSELoss({"sp": 10.0}).build(handler)
        assert loss.weights[0].item() == pytest.approx(10.0)
        assert loss.weights[1].item() == pytest.approx(1.0)

    def test_prefix_match(self):
        handler = make_scalar_handler(["t_500", "t_850", "z_500"], ["t_500", "t_850", "z_500"])
        loss = WeightedMSELoss({"t": 6.0}).build(handler)
        assert loss.weights[0].item() == pytest.approx(6.0)
        assert loss.weights[1].item() == pytest.approx(6.0)
        assert loss.weights[2].item() == pytest.approx(1.0)

    def test_unlisted_variable_defaults_to_one(self):
        handler = make_scalar_handler(["2t", "msl"], ["2t", "msl"])
        loss = WeightedMSELoss({"z": 12.0}).build(handler)
        assert (loss.weights == 1.0).all()

    def test_higher_weight_increases_loss(self):
        handler = make_scalar_handler(["t_850", "z_500"], ["t_850", "z_500"])
        pred = torch.zeros(10, 2)
        truth = torch.ones(10, 2)

        loss_uniform = WeightedMSELoss({}).build(handler)
        loss_weighted = WeightedMSELoss({"t": 10.0}).build(handler)

        assert loss_weighted(pred, truth) > loss_uniform(pred, truth)

    def test_zero_weight_ignores_variable(self):
        handler = make_scalar_handler(["t_850", "z_500"], ["t_850", "z_500"])
        pred = torch.zeros(10, 2)
        truth = torch.ones(10, 2)
        # zero weight on t_850, only z_500 contributes
        loss = WeightedMSELoss({"t": 0.0}).build(handler)
        # error only on z_500 (weight 1) -> mean over both dims -> 0.5
        assert loss(pred, truth).item() == pytest.approx(0.5)


# --- WeightedEquivariantMSELoss ---

class TestWeightedEquivariantMSELoss:
    @pytest.fixture
    def handler(self):
        return make_equivariant_handler(
            variables=["lsm", "t_850", "z_500", "sp", "10u", "10v", "u_850", "v_850"],
            out_names=["t_850", "z_500", "sp"],
            prognostic_vector={"wind_10": ["10u", "10v"], "wind_850": ["u_850", "v_850"]},
        )

    def test_scalar_weights_shape(self, handler):
        loss = WeightedEquivariantMSELoss({"z": 12.0}).build(handler)
        assert loss.scalar_weights.shape == (3,)

    def test_vector_weights_shape(self, handler):
        loss = WeightedEquivariantMSELoss({"wind": 0.6}).build(handler)
        assert loss.vector_weights.shape == (1, 2, 1)

    def test_wind_prefix_matches_all_levels(self, handler):
        loss = WeightedEquivariantMSELoss({"wind": 0.5}).build(handler)
        # vector_weights after view: shape [1, 2, 1]
        assert loss.vector_weights[0, 0, 0].item() == pytest.approx(0.5)
        assert loss.vector_weights[0, 1, 0].item() == pytest.approx(0.5)

    def test_scalar_prefix_match(self, handler):
        loss = WeightedEquivariantMSELoss({"z": 12.0, "t": 6.0}).build(handler)
        # out_names order: t_850, z_500, sp
        assert loss.scalar_weights[0].item() == pytest.approx(6.0)   # t_850
        assert loss.scalar_weights[1].item() == pytest.approx(12.0)  # z_500
        assert loss.scalar_weights[2].item() == pytest.approx(1.0)   # sp unlisted

    def test_equal_weights_matches_unweighted(self, handler):
        from equicast.model.losses import EquivariantMSELoss
        nodes, n_scalar, n_vec = 20, 3, 2
        pred = {"scalar": torch.randn(nodes, n_scalar), "vector": torch.randn(nodes, n_vec, 2)}
        truth = {"scalar": torch.randn(nodes, n_scalar), "vector": torch.randn(nodes, n_vec, 2)}

        unweighted = EquivariantMSELoss()
        weighted = WeightedEquivariantMSELoss({}).build(handler)

        assert weighted(pred, truth).item() == pytest.approx(unweighted(pred, truth).item())

    def test_higher_scalar_weight_increases_loss(self, handler):
        nodes, n_scalar, n_vec = 20, 3, 2
        pred = {"scalar": torch.zeros(nodes, n_scalar), "vector": torch.zeros(nodes, n_vec, 2)}
        truth = {"scalar": torch.ones(nodes, n_scalar), "vector": torch.zeros(nodes, n_vec, 2)}

        base = WeightedEquivariantMSELoss({}).build(handler)
        weighted = WeightedEquivariantMSELoss({"z": 12.0}).build(handler)

        assert weighted(pred, truth) > base(pred, truth)

"""Test to_raw round-trip and WeatherBenchTracker for both data handlers.

Tests:
1. EquivariantGraphDataHandler to_raw round-trip
2. EquivariantGraphDataHandler WeatherBenchTracker - perfect prediction (RMSE ~ 0)
3. EquivariantGraphDataHandler WeatherBenchTracker - zero prediction (RMSE > 0)
4. GraphDataHandler to_raw round-trip
5. GraphDataHandler WeatherBenchTracker - perfect prediction (RMSE ~ 0)
6. GraphDataHandler WeatherBenchTracker - zero prediction (RMSE > 0)
"""

import torch

from equicast.data import EquivariantGraphDataHandler, FeatureConfig
from equicast.data.data_handler import GraphDataHandler
from equicast.data.dataset.dataset import AnemoiDataset
from equicast.data.graph_provider import StaticGraphProvider
from equicast.metrics import WeatherBenchTracker

DATASET_PATH = "storage/era5-o96-2024-tail200-6h.zarr"
GRAPH_PATH = "graph/aifs-graphcast-unnormed.pt"
EQUIVARIANT_FEATURE_CONFIG_PATH = "config/features/base_equivariant.yaml"
GNN_FEATURE_CONFIG_PATH = "config/features/base.yaml"

TRACKED = ["z_500", "t_850", "q_700", "u_850", "v_850"]

graph_provider = StaticGraphProvider(graph_path=GRAPH_PATH)
dataset = AnemoiDataset(path=DATASET_PATH, graph_provider=graph_provider)
batch = dataset[0]


def section(title):
    print()
    print("=" * 55)
    print(title)
    print("=" * 55)


def test_roundtrip(dh, normalized_target, raw_original):
    recovered_raw = dh.to_raw(normalized_target)
    all_ok = True
    for name in TRACKED:
        idx = dh.name_to_index.get(name)
        if idx is None:
            print(f"  {name:<8}  not in feature config, skipping")
            continue
        max_err = (raw_original[:, idx] - recovered_raw[:, idx]).abs().max().item()
        ok = max_err < 1e-3
        all_ok = all_ok and ok
        print(f"  {name:<8}  max_err={max_err:.2e}  {'OK' if ok else 'FAIL'}")
    print(f"\n  Round-trip: {'PASS' if all_ok else 'FAIL'}")


def test_tracker(dh, tracker, normalized_target, zero_backbone_out):
    nodes = dh.nodes

    # Perfect prediction
    perfect_pred = batch["input"].clone()
    dh.update_state_with_backbone_output(perfect_pred, normalized_target)
    metrics = tracker.compute_metrics(perfect_pred, normalized_target)
    all_zero = True
    for name, val in metrics.items():
        ok = val.item() < 1e-3
        all_zero = all_zero and ok
        print(f"  {name:<30} = {val.item():.4f}  {'OK' if ok else 'FAIL'}")
    print(f"\n  Perfect pred: {'PASS' if all_zero else 'FAIL'}")

    print()

    # Zero prediction
    zero_pred = batch["input"].clone()
    dh.update_state_with_backbone_output(zero_pred, zero_backbone_out)
    metrics = tracker.compute_metrics(zero_pred, normalized_target)
    all_nonzero = True
    for name, val in metrics.items():
        ok = val.item() > 0
        all_nonzero = all_nonzero and ok
        print(f"  {name:<30} = {val.item():.4f}  {'OK' if ok else 'FAIL'}")
    print(f"\n  Zero pred: {'PASS' if all_nonzero else 'FAIL'}")


# ── EquivariantGraphDataHandler ───────────────────────────────────────────────

section("EquivariantGraphDataHandler")

eq_dh = EquivariantGraphDataHandler(
    dataset_path=DATASET_PATH,
    feature_config=FeatureConfig.from_yaml(EQUIVARIANT_FEATURE_CONFIG_PATH),
)
eq_tracker = WeatherBenchTracker(eq_dh)
eq_target_graph = batch["target"].clone()
eq_raw_original = eq_target_graph[eq_dh.nodes].data.clone()
eq_normalized_target = eq_dh.prepare_backbone_target(eq_target_graph)
eq_zero_out = {
    "scalar": torch.zeros_like(eq_normalized_target["scalar"]),
    "vector": torch.zeros_like(eq_normalized_target["vector"]),
}

section("Test 1: EquivariantGraphDataHandler to_raw round-trip")
test_roundtrip(eq_dh, eq_normalized_target, eq_raw_original)

section("Test 2-3: EquivariantGraphDataHandler WeatherBenchTracker")
test_tracker(eq_dh, eq_tracker, eq_normalized_target, eq_zero_out)

# ── GraphDataHandler ──────────────────────────────────────────────────────────

section("GraphDataHandler")

gnn_dh = GraphDataHandler(
    dataset_path=DATASET_PATH,
    feature_config=FeatureConfig.from_yaml(GNN_FEATURE_CONFIG_PATH),
)
gnn_tracker = WeatherBenchTracker(gnn_dh)
gnn_target_graph = batch["target"].clone()
gnn_raw_original = gnn_target_graph[gnn_dh.nodes].data.clone()
gnn_normalized_target = gnn_dh.prepare_backbone_target(gnn_target_graph)
gnn_zero_out = torch.zeros_like(gnn_normalized_target)

section("Test 4: GraphDataHandler to_raw round-trip")
test_roundtrip(gnn_dh, gnn_normalized_target, gnn_raw_original)

section("Test 5-6: GraphDataHandler WeatherBenchTracker")
test_tracker(gnn_dh, gnn_tracker, gnn_normalized_target, gnn_zero_out)

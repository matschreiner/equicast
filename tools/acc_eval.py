"""Plot mean ACC vs lead time averaged over all forecasts in an eval directory.

ACC = weighted spatial correlation between forecast and truth anomalies,
where anomaly = field - climatology and climatology = mean over all truth frames
at that lead time.

Usage:
    python tools/acc_eval.py experiments/.../eval_2023
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
from forecast_utils import extract_field, figures_dir, find_forecast_jsons, load_frames, load_meta, zarr_path

_WB2_LEAD_TIMES_H = [24, 72, 120, 168, 240]
_WB2_REFS = {
    "z_500":    {"IFS HRES": [0.993, 0.971, 0.934, 0.887, 0.818], "GraphCast": [0.994, 0.980, 0.955, 0.920, 0.865]},
    "t_850":    {"IFS HRES": [0.992, 0.968, 0.927, 0.876, 0.802], "GraphCast": [0.993, 0.975, 0.944, 0.901, 0.839]},
}


def weighted_acc(pred_anom, truth_anom, lat_weights):
    """Latitude-weighted spatial ACC."""
    w = lat_weights
    num   = (w * pred_anom * truth_anom).sum(axis=-1)
    denom = np.sqrt((w * pred_anom**2).sum(axis=-1) * (w * truth_anom**2).sum(axis=-1))
    return num / (denom + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir")
    parser.add_argument("--features", nargs="+", default=["z_500", "t_850"])
    parser.add_argument("--timestep-hours", type=int, default=6)
    args = parser.parse_args()

    json_files = find_forecast_jsons(args.eval_dir)
    if not json_files:
        raise FileNotFoundError(f"No forecast_*.json files found in {args.eval_dir}")

    print(f"Found {len(json_files)} forecasts")

    # --- Pass 1: compute climatology (mean truth per step) ---
    print("Pass 1: computing climatology...")
    clim   = {f: None for f in args.features}
    counts = {f: 0    for f in args.features}
    lead_times = None

    for json_path in json_files:
        try:
            meta, forecast_dir = load_meta(json_path)
            pred_ds = xr.open_zarr(zarr_path(meta, forecast_dir))
            start_idx = meta["start_idx"]
            n_input   = meta.get("n_input", 1)
            n_steps   = pred_ds.sizes["time"]

            if lead_times is None:
                lead_times = np.arange(1, n_steps + 1) * args.timestep_hours

            truth_frames, var_names = load_frames(meta["dataset"], start_idx + n_input, n_steps)

            for feature in args.features:
                truth = extract_field(truth_frames, var_names, feature)  # (n_steps, n_nodes)
                if clim[feature] is None:
                    clim[feature] = np.zeros_like(truth)
                clim[feature] += truth
                counts[feature] += 1
        except Exception as e:
            print(f"  skipping {os.path.basename(json_path)}: {e}")

    for feature in args.features:
        clim[feature] /= counts[feature]

    # --- Pass 2: compute ACC ---
    print("Pass 2: computing ACC...")
    all_acc = {f: [] for f in args.features}
    n_valid = 0

    for json_path in json_files:
        try:
            meta, forecast_dir = load_meta(json_path)
            pred_ds = xr.open_zarr(zarr_path(meta, forecast_dir))
            start_idx = meta["start_idx"]
            n_input   = meta.get("n_input", 1)
            n_steps   = pred_ds.sizes["time"]

            truth_frames, var_names = load_frames(meta["dataset"], start_idx + n_input, n_steps)
            lat = pred_ds["latitude"].values
            lat_weights = np.cos(np.radians(lat))
            lat_weights /= lat_weights.mean()

            for feature in args.features:
                pred  = extract_field(pred_ds,    None,      feature)  # (n_steps, n_nodes)
                truth = extract_field(truth_frames, var_names, feature)
                pred_anom  = pred  - clim[feature]
                truth_anom = truth - clim[feature]
                acc = weighted_acc(pred_anom, truth_anom, lat_weights)  # (n_steps,)
                all_acc[feature].append(acc)

            n_valid += 1
            if n_valid % 10 == 0:
                print(f"  {n_valid}/{len(json_files)}")

        except Exception as e:
            print(f"  skipping {os.path.basename(json_path)}: {e}")

    print(f"Computed ACC over {n_valid} forecasts")

    # --- Plot ---
    fig, axes = plt.subplots(1, len(args.features), figsize=(5 * len(args.features), 5), sharey=False)
    if len(args.features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, args.features):
        acc_arr  = np.stack(all_acc[feature])   # (n_valid, n_steps)
        mean_acc = acc_arr.mean(axis=0)
        std_acc  = acc_arr.std(axis=0)

        line, = ax.plot(lead_times, mean_acc, label="forecast", marker="o", markersize=3)
        ax.fill_between(lead_times, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2, color=line.get_color())
        ax.axhline(0.6, color="gray", linestyle=":", linewidth=1.0, label="ACC=0.6")

        for ref_name, ref_vals in _WB2_REFS.get(feature, {}).items():
            ax.plot(_WB2_LEAD_TIMES_H, ref_vals, linestyle=":", marker="s", markersize=3, label=ref_name)

        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel("ACC")
        ax.set_title(feature)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Mean ACC — {n_valid} forecasts", y=1.01)
    out_path = os.path.join(figures_dir(args.eval_dir), "rmse_plots", "acc_mean.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

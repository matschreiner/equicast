"""Plot mean RMSE vs lead time averaged over all forecasts in an eval directory.

Usage:
    python tools/rmse_eval.py experiments/.../eval_2023
"""

import argparse
import glob
import os
import sys



import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
from forecast_utils import extract_field, figures_dir, find_forecast_jsons, load_frames, load_meta, zarr_path

_WB2_LEAD_TIMES_H = [24, 72, 120, 168, 240]
_WB2_REFS = {
    "z_500":    {"IFS HRES": [41, 136, 306, 521, 809],       "GraphCast": [39, 122, 270, 459, 732]},
    "t_850":    {"IFS HRES": [0.63, 1.19, 1.84, 2.62, 3.62], "GraphCast": [0.51, 0.93, 1.54, 2.30, 3.36]},
    "q_700":    {"IFS HRES": [0.53, 0.96, 1.28, 1.55, 1.84], "GraphCast": [0.46, 0.77, 1.03, 1.28, 1.58]},
    "wind_850": {"IFS HRES": [1.60, 3.23, 5.15, 7.07, 9.13], "GraphCast": [1.40, 2.71, 4.42, 6.20, 8.27]},
}


def compute_rmse(pred_ds, truth_frames, var_names, lat_weights, feature):
    if feature.startswith("wind_"):
        level = feature[5:]
        u_name = "10u" if level == "10" else f"u_{level}"
        v_name = "10v" if level == "10" else f"v_{level}"
        pred_u = pred_ds[u_name].values
        pred_v = pred_ds[v_name].values
        u_idx, v_idx = var_names.index(u_name), var_names.index(v_name)
        truth_u = truth_frames[..., u_idx]
        truth_v = truth_frames[..., v_idx]
        sq_err = (pred_u - truth_u) ** 2 + (pred_v - truth_v) ** 2
    else:
        pred = extract_field(pred_ds, None, feature)
        truth = extract_field(truth_frames, var_names, feature)
        sq_err = (pred - truth) ** 2

    return np.sqrt((sq_err * lat_weights).mean(axis=-1))  # (n_steps,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", help="Directory containing forecast_*.json files")
    parser.add_argument("--features", nargs="+", default=["wind_850", "t_850", "z_500", "q_700"])
    parser.add_argument("--timestep-hours", type=int, default=6)
    args = parser.parse_args()

    json_files = find_forecast_jsons(args.eval_dir)
    if not json_files:
        raise FileNotFoundError(f"No forecast_*.json files found in {args.eval_dir}")

    print(f"Found {len(json_files)} forecasts")

    # Collect per-forecast RMSE arrays for mean + std
    all_rmse = {f: [] for f in args.features}
    all_pers = {f: [] for f in args.features}
    lead_times = None
    n_valid = 0

    for i, json_path in enumerate(json_files):
        try:
            meta, forecast_dir = load_meta(json_path)
            pred_ds = xr.open_zarr(zarr_path(meta, forecast_dir))

            start_idx = meta["start_idx"]
            n_input = meta.get("n_input", 1)
            n_steps = pred_ds.sizes["time"]

            truth_frames, var_names = load_frames(meta["dataset"], start_idx + n_input, n_steps)
            init_frames, _ = load_frames(meta["dataset"], start_idx + n_input - 1, 1)

            lat = pred_ds["latitude"].values
            lat_weights = np.cos(np.radians(lat))
            lat_weights /= lat_weights.mean()

            if lead_times is None:
                lead_times = np.arange(1, n_steps + 1) * args.timestep_hours

            for feature in args.features:
                rmse = compute_rmse(pred_ds, truth_frames, var_names, lat_weights, feature)
                all_rmse[feature].append(rmse)

                init = init_frames[0]  # (n_nodes, n_features)
                if feature.startswith("wind_"):
                    level = feature[5:]
                    u_name = "10u" if level == "10" else f"u_{level}"
                    v_name = "10v" if level == "10" else f"v_{level}"
                    u_idx, v_idx = var_names.index(u_name), var_names.index(v_name)
                    truth_u = truth_frames[..., u_idx]
                    truth_v = truth_frames[..., v_idx]
                    sq_p = (init[:, u_idx] - truth_u) ** 2 + (init[:, v_idx] - truth_v) ** 2
                    all_pers[feature].append(np.sqrt((sq_p * lat_weights).mean(axis=-1)))
                else:
                    truth = extract_field(truth_frames, var_names, feature)
                    init_val = extract_field(init_frames, var_names, feature)[0]
                    sq_p = (init_val - truth) ** 2
                    all_pers[feature].append(np.sqrt((sq_p * lat_weights).mean(axis=-1)))

            n_valid += 1
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(json_files)}")

        except Exception as e:
            print(f"  skipping {os.path.basename(json_path)}: {e}")

    print(f"Averaged over {n_valid} forecasts")

    fig, axes = plt.subplots(1, len(args.features), figsize=(5 * len(args.features), 5), sharey=False)
    if len(args.features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, args.features):
        rmse_arr = np.stack(all_rmse[feature])   # (n_valid, n_steps)
        pers_arr = np.stack(all_pers[feature])

        mean_rmse = rmse_arr.mean(axis=0)
        std_rmse  = rmse_arr.std(axis=0)
        mean_pers = pers_arr.mean(axis=0)
        std_pers  = pers_arr.std(axis=0)

        line, = ax.plot(lead_times, mean_rmse, label="forecast", marker="o", markersize=3)
        ax.fill_between(lead_times, mean_rmse - std_rmse, mean_rmse + std_rmse, alpha=0.2, color=line.get_color())

        line_p, = ax.plot(lead_times, mean_pers, label="persistence", linestyle="--", marker="o", markersize=3)
        ax.fill_between(lead_times, mean_pers - std_pers, mean_pers + std_pers, alpha=0.15, color=line_p.get_color())
        for ref_name, ref_vals in _WB2_REFS.get(feature, {}).items():
            ax.plot(_WB2_LEAD_TIMES_H, ref_vals, linestyle=":", marker="s", markersize=3, label=ref_name)

        units = "g/kg" if feature.startswith("q") else None
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel(f"RMSE ({units})" if units else "RMSE")
        ax.set_title(feature)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Mean RMSE — {n_valid} forecasts", y=1.01)
    output_path = os.path.join(figures_dir(args.eval_dir), "rmse_mean.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

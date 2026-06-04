"""Plot RMSE vs forecast lead time for selected variables.

Usage:
    python tools/rmse_forecast.py experiments/07_painn_big/forecasts/f1b9.../2026-05-27_14:30/forecast.json
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from forecast_utils import extract_field, figures_dir, load_frames, load_meta, zarr_path

# WeatherBench2 reference values at lead times [1, 3, 5, 7, 10] days (in hours: [24, 72, 120, 168, 240])
_WB2_LEAD_TIMES_H = [24, 72, 120, 168, 240]
_WB2_REFS = {
    "z_500":    {"IFS HRES": [41, 136, 306, 521, 809],     "GraphCast": [39, 122, 270, 459, 732]},
    "t_850":    {"IFS HRES": [0.63, 1.19, 1.84, 2.62, 3.62], "GraphCast": [0.51, 0.93, 1.54, 2.30, 3.36]},
    "q_700":    {"IFS HRES": [0.53, 0.96, 1.28, 1.55, 1.84], "GraphCast": [0.46, 0.77, 1.03, 1.28, 1.58]},
    "wind_850": {"IFS HRES": [1.60, 3.23, 5.15, 7.07, 9.13], "GraphCast": [1.40, 2.71, 4.42, 6.20, 8.27]},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast", help="Path to forecast.json or forecast directory")
    parser.add_argument("--features", nargs="+", default=["wind_850", "t_850", "z_500", "q_700"])
    parser.add_argument("--timestep-hours", type=int, default=6)
    args = parser.parse_args()

    meta, forecast_dir = load_meta(args.forecast)
    pred_ds = xr.open_zarr(zarr_path(meta, forecast_dir))

    start_idx = meta["start_idx"]
    n_input = meta.get("n_input", 1)
    n_steps = pred_ds.dims["time"]
    lead_times = np.arange(1, n_steps + 1) * args.timestep_hours

    truth_frames, var_names = load_frames(meta["dataset"], start_idx + n_input, n_steps)
    init_frames, _ = load_frames(meta["dataset"], start_idx + n_input - 1, 1)

    fig, axes = plt.subplots(1, len(args.features), figsize=(5 * len(args.features), 5), sharey=False)
    if len(args.features) == 1:
        axes = [axes]

    lat_rad = np.radians(pred_ds["latitude"].values)
    lat_weights = np.cos(lat_rad)
    lat_weights /= lat_weights.mean()

    for ax, feature in zip(axes, args.features):
        if feature.startswith("wind_"):
            level = feature[5:]
            u_name, v_name = ("10u", "10v") if level == "10" else (f"u_{level}", f"v_{level}")
            pred_u = pred_ds[u_name].values
            pred_v = pred_ds[v_name].values
            u_idx, v_idx = var_names.index(u_name), var_names.index(v_name)
            truth_u, truth_v = truth_frames[..., u_idx], truth_frames[..., v_idx]
            init_u, init_v = init_frames[0, :, u_idx], init_frames[0, :, v_idx]
            sq_err = (pred_u - truth_u) ** 2 + (pred_v - truth_v) ** 2
            sq_pers = (init_u - truth_u) ** 2 + (init_v - truth_v) ** 2
            rmse = np.sqrt((sq_err * lat_weights).mean(axis=-1))
            persistence = np.sqrt((sq_pers * lat_weights).mean(axis=-1))
        else:
            pred = extract_field(pred_ds, None, feature)
            truth = extract_field(truth_frames, var_names, feature)
            init = extract_field(init_frames, var_names, feature)[0]
            rmse = np.sqrt((((pred - truth) ** 2) * lat_weights).mean(axis=-1))
            persistence = np.sqrt((((init - truth) ** 2) * lat_weights).mean(axis=-1))

        ax.plot(lead_times, rmse, label="forecast", marker="o", markersize=3)
        ax.plot(lead_times, persistence, label="persistence", linestyle="--", marker="o", markersize=3)
        for ref_name, ref_vals in _WB2_REFS.get(feature, {}).items():
            ax.plot(_WB2_LEAD_TIMES_H, ref_vals, linestyle=":", marker="s", markersize=3, label=ref_name)
        units = "g/kg" if feature.startswith("q") else None
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel(f"RMSE ({units})" if units else "RMSE")
        ax.set_title(feature)
        ax.legend()
        ax.grid(True, alpha=0.3)

    output_path = os.path.join(figures_dir(forecast_dir), "rmse.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

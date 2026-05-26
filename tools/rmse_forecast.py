"""Plot RMSE vs forecast lead time for selected variables.

Usage:
    python tools/rmse_forecast.py experiments/07_painn_big/forecasts/47a62ecbb4ff46c39a33575ef708a678
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def load_field(ds, feature):
    if feature.startswith("wind_"):
        level = feature[5:]
        u, v = (ds["10u"], ds["10v"]) if level == "10" else (ds[f"u_{level}"], ds[f"v_{level}"])
        return np.sqrt(u.values ** 2 + v.values ** 2)
    values = ds[feature].values
    if feature.startswith("q"):
        values = values * 1000  # kg/kg -> g/kg
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast_dir")
    parser.add_argument("--features", nargs="+", default=["wind_850", "t_850", "z_500", "q_700"])
    parser.add_argument("--timestep-hours", type=int, default=6)
    args = parser.parse_args()

    pred_ds = xr.open_zarr(os.path.join(args.forecast_dir, "forecast.zarr"))
    truth_ds = xr.open_zarr(os.path.join(args.forecast_dir, "target_forecast.zarr"))
    init_ds = xr.open_zarr(os.path.join(args.forecast_dir, "initial_state.zarr"))

    n_steps = pred_ds.dims["time"]
    lead_times = np.arange(1, n_steps + 1) * args.timestep_hours

    fig, axes = plt.subplots(1, len(args.features), figsize=(5 * len(args.features), 5), sharey=False)
    if len(args.features) == 1:
        axes = [axes]

    for ax, feature in zip(axes, args.features):
        pred = load_field(pred_ds, feature)
        truth = load_field(truth_ds, feature)
        init = load_field(init_ds, feature)[0]

        rmse = np.sqrt(((pred - truth) ** 2).mean(axis=-1))
        persistence = np.sqrt(((init - truth) ** 2).mean(axis=-1))

        ax.plot(lead_times, rmse, label="forecast", marker="o", markersize=3)
        ax.plot(lead_times, persistence, label="persistence", linestyle="--", marker="o", markersize=3)
        units = "g/kg" if feature.startswith("q") else None
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel(f"RMSE ({units})" if units else "RMSE")
        ax.set_title(feature)
        ax.legend()
        ax.grid(True, alpha=0.3)

    output_path = os.path.join(args.forecast_dir, "rmse.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

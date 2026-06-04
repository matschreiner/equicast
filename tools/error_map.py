"""Plot spatial error map for a variable at a given forecast step.

Usage:
    python tools/error_map.py forecast.json
    python tools/error_map.py forecast.json --step 1 --var t_850
    python tools/error_map.py forecast.json --step 1 --var z_500
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import xarray as xr

from forecast_utils import figures_dir, load_frames, load_meta, zarr_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast", help="Path to forecast.json or forecast directory")
    parser.add_argument("--step", type=int, default=1, help="Forecast step (1-indexed, default: 1)")
    parser.add_argument("--var", type=str, default="t_850", help="Variable name (default: t_850)")
    args = parser.parse_args()

    meta, forecast_dir = load_meta(args.forecast)
    pred_ds = xr.open_zarr(zarr_path(meta, forecast_dir))

    start_idx = meta["start_idx"]
    n_input = meta.get("n_input", 1)
    step = args.step - 1  # 0-indexed

    truth_frames, var_names = load_frames(meta["dataset"], start_idx + n_input + step, 1)
    truth = truth_frames[0]  # (n_nodes, n_features)

    var = args.var
    pred_vals = pred_ds[var].values[step]
    t_idx = var_names.index(var)
    truth_vals = truth[:, t_idx]
    error = pred_vals - truth_vals

    lat = pred_ds["latitude"].values
    lon = pred_ds["longitude"].values
    triang = mtri.Triangulation(lon, lat)

    emax = np.percentile(np.abs(error), 95)
    shared_levels = np.linspace(
        min(truth_vals.min(), pred_vals.min()),
        max(truth_vals.max(), pred_vals.max()),
        51,
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    cf = ax.tricontourf(triang, error, levels=32, cmap="RdBu_r", vmin=-emax, vmax=emax)
    plt.colorbar(cf, ax=ax)
    ax.tricontour(triang, truth_vals, levels=shared_levels, colors="black", linewidths=1.2, linestyles="solid")
    ax.tricontour(triang, pred_vals, levels=shared_levels, colors=[(1, 0, 0, 0.5)], linewidths=1.2, linestyles="dashed")
    ax.plot([], [], color="black", linestyle="solid", label="truth")
    ax.plot([], [], color="red", linestyle="dashed", label="forecast")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"{var} error — step {args.step} ({args.step * 6}h)")

    output_path = os.path.join(figures_dir(forecast_dir), f"error_map_{var}_step{args.step}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

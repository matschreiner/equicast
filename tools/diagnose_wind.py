"""Plot wind speed in predictions vs ground truth.

Shows global mean wind speed per timestep for pred and truth,
plus wind speed at a few sample nodes.

Usage:
    python scripts/diagnose_wind.py forecasts/80a28951aa9941578184d78d870e663e/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast_dir")
    parser.add_argument("--level", default="500", help="Pressure level (default: 500)")
    args = parser.parse_args()

    pred_ds = xr.open_zarr(os.path.join(args.forecast_dir, "predictions.zarr"))
    truth_ds = xr.open_zarr(os.path.join(args.forecast_dir, "input_timeseries.zarr"))

    u_var = f"u_{args.level}"
    v_var = f"v_{args.level}"

    u_pred = pred_ds[u_var].values
    v_pred = pred_ds[v_var].values
    u_truth = truth_ds[u_var].values
    v_truth = truth_ds[v_var].values

    n = min(len(u_pred), len(u_truth))
    u_pred, v_pred = u_pred[:n], v_pred[:n]
    u_truth, v_truth = u_truth[:n], v_truth[:n]

    ws_pred = np.sqrt(u_pred**2 + v_pred**2)
    ws_truth = np.sqrt(u_truth**2 + v_truth**2)

    hours = np.arange(n) * 6

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # 1. Global mean wind speed
    ax = axes[0]
    ax.plot(hours, ws_truth.mean(axis=1), label="truth", linewidth=2)
    ax.plot(hours, ws_pred.mean(axis=1), label="pred", linewidth=2, alpha=0.8)
    ax.set_ylabel("Wind speed [m/s]")
    ax.set_title(f"Global mean wind speed at {args.level} hPa")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Global mean u and v separately
    ax = axes[1]
    ax.plot(hours, u_truth.mean(axis=1), label="u truth", linewidth=1.5, color="C0")
    ax.plot(hours, u_pred.mean(axis=1), label="u pred", linewidth=1.5, color="C0", linestyle="--", alpha=0.8)
    ax.plot(hours, v_truth.mean(axis=1), label="v truth", linewidth=1.5, color="C1")
    ax.plot(hours, v_pred.mean(axis=1), label="v pred", linewidth=1.5, color="C1", linestyle="--", alpha=0.8)
    ax.set_ylabel("Wind component [m/s]")
    ax.set_title(f"Global mean u/v at {args.level} hPa")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. Absolute errors (global mean)
    ax = axes[2]
    ae_u = np.abs(u_pred - u_truth).mean(axis=1)
    ae_v = np.abs(v_pred - v_truth).mean(axis=1)
    ae_ws = np.abs(ws_pred - ws_truth).mean(axis=1)
    ax.plot(hours, ae_ws, label="wind speed MAE", linewidth=2)
    ax.plot(hours, ae_u, label="u MAE", linewidth=1.5, alpha=0.8)
    ax.plot(hours, ae_v, label="v MAE", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("MAE [m/s]")
    ax.set_title(f"Mean absolute error at {args.level} hPa")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. RMSE of wind speed and components
    ax = axes[3]
    rmse_ws = np.sqrt(((ws_pred - ws_truth) ** 2).mean(axis=1))
    rmse_u = np.sqrt(((u_pred - u_truth) ** 2).mean(axis=1))
    rmse_v = np.sqrt(((v_pred - v_truth) ** 2).mean(axis=1))
    ax.plot(hours, rmse_ws, label="wind speed RMSE", linewidth=2)
    ax.plot(hours, rmse_u, label="u RMSE", linewidth=1.5, alpha=0.8)
    ax.plot(hours, rmse_v, label="v RMSE", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Lead time (hours)")
    ax.set_ylabel("RMSE [m/s]")
    ax.set_title(f"Wind RMSE at {args.level} hPa")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(args.forecast_dir, f"wind_diagnostic_{args.level}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()

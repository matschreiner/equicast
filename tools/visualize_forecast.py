"""Visualize forecast predictions vs ground truth from a forecast directory.

Usage:
    python scripts/visualize_forecast.py forecasts/80a28951aa99/ 2t
"""

import argparse
import os

import numpy as np
import xarray as xr

from equicast.visualization import make_comparison_video


def main():
    parser = argparse.ArgumentParser(description="Visualize forecast comparison")
    parser.add_argument("forecast_dir", help="Path to forecast directory")
    parser.add_argument("feature", help="Feature name to visualize")
    args = parser.parse_args()

    pred_ds = xr.open_dataset(os.path.join(args.forecast_dir, "forecast.zarr"))
    truth_ds = xr.open_dataset(os.path.join(args.forecast_dir, "target_forecast.zarr"))

    lat = pred_ds["latitude"].values
    lon = pred_ds["longitude"].values
    latlon = np.stack([np.radians(lat), np.radians(lon)], axis=-1)

    if args.feature.startswith("wind_"):
        level = args.feature[5:]
        u_name, v_name = (f"10u", f"10v") if level == "10" else (f"u_{level}", f"v_{level}")
        pred = np.sqrt(pred_ds[u_name].values ** 2 + pred_ds[v_name].values ** 2)
        truth = np.sqrt(truth_ds[u_name].values ** 2 + truth_ds[v_name].values ** 2)
    else:
        pred = pred_ds[args.feature].values
        truth = truth_ds[args.feature].values
    n = min(len(pred), len(truth))

    output_path = os.path.join(args.forecast_dir, f"{args.feature}.mp4")
    make_comparison_video(
        predictions=pred[:n],
        targets=truth[:n],
        latlon=latlon,
        output_path=output_path,
        title_template=f"{args.feature} – Frame {{frame}}",
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

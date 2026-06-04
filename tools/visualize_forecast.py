"""Visualize forecast predictions vs ground truth.

Usage:
    python tools/visualize_forecast.py forecast.json wind_850
    python tools/visualize_forecast.py <forecast_dir> wind_850
"""

import argparse
import os

import numpy as np
import xarray as xr

from equicast.visualization import make_comparison_video
from forecast_utils import extract_field, figures_dir, load_frames, load_meta, zarr_path


def main():
    parser = argparse.ArgumentParser(description="Visualize forecast comparison")
    parser.add_argument("forecast", help="Path to forecast.json or forecast directory")
    parser.add_argument("feature", help="Feature name to visualize")
    args = parser.parse_args()

    meta, forecast_dir = load_meta(args.forecast)
    pred_ds = xr.open_zarr(zarr_path(meta, forecast_dir))

    start_idx = meta["start_idx"]
    n_input = meta.get("n_input", 1)
    n_steps = pred_ds.sizes["time"]

    truth_frames, var_names = load_frames(meta["dataset"], start_idx + n_input, n_steps)

    lat = pred_ds["latitude"].values
    lon = pred_ds["longitude"].values
    latlon = np.stack([np.radians(lat), np.radians(lon)], axis=-1)

    pred = extract_field(pred_ds, None, args.feature)
    truth = extract_field(truth_frames, var_names, args.feature)

    output_path = os.path.join(figures_dir(os.path.dirname(forecast_dir)), f"{args.feature}.mp4")
    make_comparison_video(
        predictions=pred,
        targets=truth,
        latlon=latlon,
        output_path=output_path,
        title_template=f"{args.feature} – Frame {{frame}}",
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

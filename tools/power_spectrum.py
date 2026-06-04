"""Plot spherical harmonic power spectrum: forecast vs ground truth.

Usage:
    python tools/power_spectrum.py forecast.json
    python tools/power_spectrum.py forecast.json --steps 1 4 8 --var t_850
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh
import xarray as xr
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(__file__))
from forecast_utils import figures_dir, load_frames, load_meta, zarr_path


def to_regular_grid(values, lon_src, lat_src, nlat=180):
    """Interpolate unstructured (lon, lat) field to regular lat/lon grid."""
    nlon = nlat * 2
    lat_grid = np.linspace(-90, 90, nlat)
    lon_grid = np.linspace(0, 360, nlon, endpoint=False)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
    regular = griddata((lon_src, lat_src), values, (lon2d, lat2d), method="linear")
    # fill NaNs at poles with nearest
    mask = np.isnan(regular)
    if mask.any():
        regular[mask] = griddata(
            (lon_src, lat_src), values, (lon2d[mask], lat2d[mask]), method="nearest"
        )
    return regular  # (nlat, nlon)


def power_spectrum(field_regular):
    """Compute power spectrum from a regular (nlat, nlon) field."""
    coeffs = pysh.expand.SHExpandDH(field_regular, sampling=2)
    return pysh.spectralanalysis.spectrum(coeffs)  # (lmax+1,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast", help="Path to forecast.json or forecast directory")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 4, 8, 16, 40],
                        help="Forecast steps to plot (1-indexed)")
    parser.add_argument("--var", type=str, default="t_850")
    parser.add_argument("--nlat", type=int, default=720,
                        help="Regular grid latitude points (must be even). nlat=720 → lmax≈359 (~112 km).")
    args = parser.parse_args()

    meta, forecast_dir = load_meta(args.forecast)
    pred_ds = xr.open_zarr(zarr_path(meta, forecast_dir))

    start_idx = meta["start_idx"]
    n_input = meta.get("n_input", 1)

    lat = pred_ds["latitude"].values
    lon = pred_ds["longitude"].values

    # Clip steps to available range
    n_steps = pred_ds.sizes["time"]
    steps = [s for s in args.steps if 1 <= s <= n_steps]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(steps)))

    _R_EARTH_KM = 6371.0

    for color, step in zip(colors, steps):
        step0 = step - 1  # 0-indexed

        truth_frames, var_names = load_frames(meta["dataset"], start_idx + n_input + step0, 1)
        t_idx = var_names.index(args.var)
        truth_vals = truth_frames[0, :, t_idx]
        pred_vals = pred_ds[args.var].values[step0]

        truth_grid = to_regular_grid(truth_vals, lon, lat, nlat=args.nlat)
        pred_grid = to_regular_grid(pred_vals, lon, lat, nlat=args.nlat)

        ps_truth = power_spectrum(truth_grid)[1:]  # skip l=0
        ps_pred  = power_spectrum(pred_grid)[1:]
        l = np.arange(1, len(ps_truth) + 1)
        wl_km = 2 * np.pi * _R_EARTH_KM / l

        ax.plot(wl_km, ps_truth, color=color, linestyle="-",  linewidth=1.5, label=f"truth  {step*6}h")
        ax.plot(wl_km, ps_pred,  color=color, linestyle="--", linewidth=1.5, label=f"pred   {step*6}h")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength (km)")
    ax.set_ylabel("Power")
    ax.set_title(f"Power spectrum — {args.var}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(False)

    output_path = os.path.join(figures_dir(os.path.dirname(forecast_dir)), f"power_spectrum_{args.var}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

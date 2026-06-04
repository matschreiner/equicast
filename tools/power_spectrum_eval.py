"""Mean power spectrum over all eval forecasts at WB2 lead times.

Usage:
    python tools/power_spectrum_eval.py experiments/.../eval_2023 --var t_850
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyshtools as pysh
import xarray as xr
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(__file__))
from forecast_utils import figures_dir, find_forecast_jsons, load_frames, load_meta, zarr_path

_WB2_LEAD_TIMES_H = [24, 72, 120, 168, 240]


def to_regular_grid(values, lon_src, lat_src, nlat=180):
    nlon = nlat * 2
    lat_grid = np.linspace(-90, 90, nlat)
    lon_grid = np.linspace(0, 360, nlon, endpoint=False)
    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
    regular = griddata((lon_src, lat_src), values, (lon2d, lat2d), method="linear")
    mask = np.isnan(regular)
    if mask.any():
        regular[mask] = griddata(
            (lon_src, lat_src), values, (lon2d[mask], lat2d[mask]), method="nearest"
        )
    return regular


def power_spectrum(field_regular):
    coeffs = pysh.expand.SHExpandDH(field_regular, sampling=2)
    return pysh.spectralanalysis.spectrum(coeffs)


_R = 6371.0


def _plot_one(pred_all, truth_all, lead_times_h, lt_idx, var, n_valid, out_path):
    lt = lead_times_h[lt_idx]
    lmax = pred_all.shape[2]
    l = np.arange(1, lmax + 1)
    wl_km = 2 * np.pi * _R / l

    mean_pred  = pred_all[:, lt_idx].mean(axis=0)
    mean_truth = truth_all[:, lt_idx].mean(axis=0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(wl_km, mean_pred,  color="steelblue", linestyle="-",  linewidth=1.5, label="pred")
    ax1.plot(wl_km, mean_truth, color="black",     linestyle="--", linewidth=1.5, label="truth")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.invert_xaxis()
    ax1.set_ylabel("Power")
    ax1.set_title(f"{var} — {lt}h  (n={n_valid})")
    ax1.legend(); ax1.grid(False)

    ax2.plot(wl_km, mean_pred / mean_truth, color="steelblue", linewidth=1.5)
    ax2.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_xscale("log")
    ax2.set_xlabel("Wavelength (km)"); ax2.set_ylabel("pred / truth")
    ax2.grid(False)

    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", help="Directory containing forecast_*.json files")
    parser.add_argument("--var", type=str, default="t_850")
    parser.add_argument("--vars", nargs="+", help="Multiple variables — generates one plot per (var, leadtime)")
    parser.add_argument("--lead-times", type=int, nargs="+", help="Lead times to plot in hours (default: all WB2)")
    parser.add_argument("--subdir", type=str, default=None, help="Subdirectory inside figures/ to save plots")
    parser.add_argument("--timestep-hours", type=int, default=6)
    args = parser.parse_args()

    var_list = args.vars if args.vars else [args.var]

    for var in var_list:
        processed_path = os.path.join(args.eval_dir, "forecasts_processed", f"{var}_spectra.npz")
        if not os.path.exists(processed_path):
            print(f"No precomputed spectra for {var}, skipping")
            continue

        print(f"Loading {processed_path}")
        data = np.load(processed_path)
        pred_all     = data["pred"]    # (n_forecasts, n_lt, lmax)
        truth_all    = data["truth"]
        lead_times_h = data["lead_times_h"].tolist()
        n_valid      = pred_all.shape[0]
        base_figs = figures_dir(args.eval_dir)
        figs = os.path.join(base_figs, args.subdir) if args.subdir else base_figs
        os.makedirs(figs, exist_ok=True)

        selected_lts = set(args.lead_times) if args.lead_times else set(lead_times_h)

        # One plot per lead time
        for j, lt in enumerate(lead_times_h):
            if lt not in selected_lts:
                continue
            out_path = os.path.join(figs, f"power_spectrum_{var}_{lt}h.png")
            _plot_one(pred_all, truth_all, lead_times_h, j, var, n_valid, out_path)
            print(f"  Saved {out_path}")

    return


if __name__ == "__main__":
    main()

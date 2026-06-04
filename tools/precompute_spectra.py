"""Precompute power spectra for all forecasts in an eval directory.

Saves results to <eval_dir>/forecasts_processed/<var>_spectra.npz with arrays:
  pred  : (n_forecasts, n_lead_times, lmax+1)
  truth : (n_forecasts, n_lead_times, lmax+1)
  lead_times_h : (n_lead_times,)

Usage:
    python tools/precompute_spectra.py experiments/.../eval_2023
    python tools/precompute_spectra.py experiments/.../eval_2023 --vars t_850 z_500 wind_850 q_700
"""

import argparse
import os
import sys

import numpy as np
import pyshtools as pysh
import xarray as xr
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(__file__))
from forecast_utils import find_forecast_jsons, load_frames, load_meta, zarr_path

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


def get_field(pred_ds, truth_frames, var_names, var, step0):
    if var.startswith("wind_"):
        level = var[5:]
        u_name = "10u" if level == "10" else f"u_{level}"
        v_name = "10v" if level == "10" else f"v_{level}"
        u_idx, v_idx = var_names.index(u_name), var_names.index(v_name)
        truth_vals = truth_frames[0, :, u_idx] ** 2 + truth_frames[0, :, v_idx] ** 2
        pred_vals  = pred_ds[u_name].values[step0] ** 2 + pred_ds[v_name].values[step0] ** 2
    else:
        t_idx = var_names.index(var)
        truth_vals = truth_frames[0, :, t_idx]
        pred_vals  = pred_ds[var].values[step0]
    return pred_vals, truth_vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir")
    parser.add_argument("--vars", nargs="+", default=["t_850", "z_500", "wind_850", "q_700"])
    parser.add_argument("--timestep-hours", type=int, default=6)
    args = parser.parse_args()

    json_files = find_forecast_jsons(args.eval_dir)
    if not json_files:
        raise FileNotFoundError(f"No forecast_*.json files in {args.eval_dir}")

    steps = [lt // args.timestep_hours for lt in _WB2_LEAD_TIMES_H]
    out_dir = os.path.join(args.eval_dir, "forecasts_processed")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Found {len(json_files)} forecasts — vars: {args.vars}")

    # spectra[var] = {"pred": list of (n_lead_times, lmax+1), "truth": ...}
    spectra = {v: {"pred": [], "truth": []} for v in args.vars}
    lmax = None

    for i, json_path in enumerate(json_files):
        try:
            meta, forecast_dir = load_meta(json_path)
            pred_ds = xr.open_zarr(zarr_path(meta, forecast_dir))

            start_idx = meta["start_idx"]
            n_input = meta.get("n_input", 1)
            lat = pred_ds["latitude"].values
            lon = pred_ds["longitude"].values

            ps_pred_all  = {v: [] for v in args.vars}
            ps_truth_all = {v: [] for v in args.vars}

            for step, lt in zip(steps, _WB2_LEAD_TIMES_H):
                step0 = step - 1
                truth_frames, var_names = load_frames(
                    meta["dataset"], start_idx + n_input + step0, 1
                )
                for var in args.vars:
                    pred_vals, truth_vals = get_field(pred_ds, truth_frames, var_names, var, step0)
                    ps_p = power_spectrum(to_regular_grid(pred_vals,  lon, lat))
                    ps_t = power_spectrum(to_regular_grid(truth_vals, lon, lat))
                    ps_pred_all[var].append(ps_p)
                    ps_truth_all[var].append(ps_t)
                    if lmax is None:
                        lmax = len(ps_p)

            for var in args.vars:
                spectra[var]["pred"].append(np.stack(ps_pred_all[var]))   # (n_lt, lmax)
                spectra[var]["truth"].append(np.stack(ps_truth_all[var]))

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(json_files)}")

        except Exception as e:
            print(f"  skipping {os.path.basename(json_path)}: {e}")

    for var in args.vars:
        if not spectra[var]["pred"]:
            continue
        out_path = os.path.join(out_dir, f"{var}_spectra.npz")
        np.savez(
            out_path,
            pred=np.stack(spectra[var]["pred"]),    # (n_forecasts, n_lt, lmax)
            truth=np.stack(spectra[var]["truth"]),
            lead_times_h=np.array(_WB2_LEAD_TIMES_H),
        )
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

"""Shared utilities for forecast visualization scripts."""

import json
import os

import numpy as np
import xarray as xr
from anemoi.datasets import open_dataset


def load_meta(path):
    """Load forecast metadata from a forecast.json or forecast directory."""
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f), os.path.dirname(path)
    with open(os.path.join(path, "forecast.json")) as f:
        meta = json.load(f)
    return meta, meta.get("output_dir", path)


def zarr_path(meta, forecast_dir):
    """Return path to the forecast zarr for this meta (handles batch naming)."""
    start_idx = meta.get("start_idx", 0)
    prefix = f"forecast_{start_idx:04d}" if meta.get("batch") else "forecast"
    return os.path.join(forecast_dir, f"{prefix}.zarr")


def figures_dir(output_dir):
    """Return (and create) the figures subdirectory for an output directory."""
    path = os.path.join(output_dir, "figures")
    os.makedirs(path, exist_ok=True)
    return path


def find_forecast_jsons(eval_dir):
    """Find forecast_*.json files in eval_dir or its forecasts/ subdirectory."""
    import glob
    subdir = os.path.join(eval_dir, "forecasts")
    if os.path.isdir(subdir):
        files = sorted(glob.glob(os.path.join(subdir, "forecast_*.json")))
        if files:
            return files
    return sorted(glob.glob(os.path.join(eval_dir, "forecast_*.json")))


def load_frames(dataset_path, start, n_frames):
    """Load n_frames from an anemoi dataset starting at index start.

    Returns (frames, variable_names) where frames has shape (n_frames, n_nodes, n_features).
    """
    ds = open_dataset(dataset_path)
    variable_names = list(ds.variables)
    raw = ds[start : start + n_frames]  # (n_frames, n_features, 1, n_nodes)
    frames = raw[:, :, 0, :].transpose(0, 2, 1)  # (n_frames, n_nodes, n_features)
    return frames, variable_names


def extract_field(frames_or_ds, variable_names, feature):
    """Extract a field (possibly wind speed) from frames or xarray dataset."""
    if isinstance(frames_or_ds, xr.Dataset):
        ds = frames_or_ds
        if feature.startswith("wind_"):
            level = feature[5:]
            u_name, v_name = ("10u", "10v") if level == "10" else (f"u_{level}", f"v_{level}")
            values = np.sqrt(ds[u_name].values ** 2 + ds[v_name].values ** 2)
        else:
            values = ds[feature].values
    else:
        # numpy frames: (n_frames, n_nodes, n_features)
        frames = frames_or_ds
        if feature.startswith("wind_"):
            level = feature[5:]
            u_name, v_name = ("10u", "10v") if level == "10" else (f"u_{level}", f"v_{level}")
            u_idx = variable_names.index(u_name)
            v_idx = variable_names.index(v_name)
            values = np.sqrt(frames[..., u_idx] ** 2 + frames[..., v_idx] ** 2)
        else:
            idx = variable_names.index(feature)
            values = frames[..., idx]

    if feature.startswith("q"):
        values = values * 1000  # kg/kg -> g/kg
    return values

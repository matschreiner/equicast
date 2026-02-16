"""Validate forecast predictions against ground truth.

Computes global mean RMSE and bias per variable and lead time,
following WeatherBench conventions. Saves results as CSV and plots
in the forecast directory.

Usage:
    python scripts/validate_forecast.py forecasts/80a28951aa9941578184d78d870e663e/
"""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def rmse(pred, truth):
    """RMSE over the node dimension."""
    return np.sqrt(((pred - truth) ** 2).mean(dim="node"))


def bias(pred, truth):
    """Bias over the node dimension."""
    return (pred - truth).mean(dim="node")


def group_variables(variables):
    """Group variables by type (e.g. t_500 -> t, z_850 -> z)."""
    groups = defaultdict(list)
    for var in variables:
        parts = var.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            groups[parts[0]].append(var)
        else:
            groups[var].append(var)
    return dict(groups)


def plot_rmse_vs_leadtime(df, groups, forecast_dir):
    """Plot RMSE vs lead time, one subplot per variable group."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    sorted_groups = sorted(groups.keys())
    for i, group_name in enumerate(sorted_groups):
        if i >= len(axes):
            break
        ax = axes[i]
        members = sorted(groups[group_name])
        for var in members:
            var_df = df[df["variable"] == var]
            label = var.split("_")[1] if "_" in var and var.split("_")[1].isdigit() else var
            ax.plot(var_df["lead_time"], var_df["rmse"], label=label)
        ax.set_title(group_name.upper())
        ax.set_xlabel("Lead time (steps)")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    for i in range(len(sorted_groups), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("RMSE vs Lead Time", fontsize=14)
    fig.tight_layout()
    path = os.path.join(forecast_dir, "rmse_vs_leadtime.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_mean_rmse_bar(df, forecast_dir):
    """Bar chart of mean RMSE per variable."""
    summary = df.groupby("variable")["rmse"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(14, 8))
    summary.plot.barh(ax=ax)
    ax.set_xlabel("Mean RMSE (global mean)")
    ax.set_title("Mean RMSE per Variable")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(forecast_dir, "mean_rmse.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_bias_vs_leadtime(df, groups, forecast_dir):
    """Plot bias vs lead time for key variables."""
    key_vars = ["z_500", "t_850", "2t", "msl", "u_250", "q_700"]
    key_vars = [v for v in key_vars if v in df["variable"].values]

    if not key_vars:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, var in enumerate(key_vars):
        if i >= len(axes):
            break
        ax = axes[i]
        var_df = df[df["variable"] == var]
        ax.plot(var_df["lead_time"], var_df["bias"], color="C0")
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.fill_between(var_df["lead_time"], 0, var_df["bias"], alpha=0.2)
        ax.set_title(var)
        ax.set_xlabel("Lead time (steps)")
        ax.set_ylabel("Bias")
        ax.grid(True, alpha=0.3)

    for i in range(len(key_vars), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Bias vs Lead Time", fontsize=14)
    fig.tight_layout()
    path = os.path.join(forecast_dir, "bias_vs_leadtime.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_headline_rmse(df, forecast_dir):
    """Plot z_500 and t_850 RMSE vs lead time (first 28 steps)."""
    headline = {"z_500": ("Z500 RMSE [m²/s²]", "z_500.png"),
                "t_850": ("T850 RMSE [K]", "t_850.png"),
                "u_500": ("U500 RMSE [m/s]", "u_500.png"),
                "v_500": ("V500 RMSE [m/s]", "v_500.png")}

    for var, (ylabel, filename) in headline.items():
        var_df = df[(df["variable"] == var) & (df["lead_time"] < 28)]
        if var_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        hours = var_df["lead_time"] * 6
        ax.plot(hours, var_df["rmse"], linewidth=2)
        ax.set_xlabel("Lead time (hours)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{var} — RMSE")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(forecast_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def validate(forecast_dir):
    pred_ds = xr.open_zarr(os.path.join(forecast_dir, "predictions.zarr"))
    truth_ds = xr.open_zarr(os.path.join(forecast_dir, "input_timeseries.zarr"))

    variables = list(pred_ds.data_vars)
    n_times = min(len(pred_ds.time), len(truth_ds.time))

    pred_ds = pred_ds.isel(time=slice(0, n_times)).compute()
    truth_ds = truth_ds.isel(time=slice(0, n_times)).compute()

    rows = []
    for var in variables:
        pred = pred_ds[var]
        truth = truth_ds[var]

        rmse_per_step = rmse(pred, truth).values
        bias_per_step = bias(pred, truth).values

        for t in range(n_times):
            rows.append({
                "variable": var,
                "lead_time": t,
                "rmse": rmse_per_step[t],
                "bias": bias_per_step[t],
            })

    df = pd.DataFrame(rows)

    output_path = os.path.join(forecast_dir, "metrics.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved metrics to {output_path}")

    # Print summary
    summary = df.groupby("variable")["rmse"].mean().sort_values(ascending=False)
    print("\nMean RMSE per variable (global mean):")
    print(summary.to_string())

    # Plots
    groups = group_variables(variables)
    plot_rmse_vs_leadtime(df, groups, forecast_dir)
    plot_mean_rmse_bar(df, forecast_dir)
    plot_bias_vs_leadtime(df, groups, forecast_dir)
    plot_headline_rmse(df, forecast_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate forecast")
    parser.add_argument("forecast_dir", help="Path to forecast directory")
    args = parser.parse_args()
    validate(args.forecast_dir)

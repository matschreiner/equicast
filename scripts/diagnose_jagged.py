"""Diagnose jagged wind RMSE in forecasts.

Checks:
1. Is the RMSE truly oscillating (goes down after going up)?
2. Does shifting predictions by ±1 step reduce error? (alignment bug)
3. What does the actual predicted vs true wind look like at a single node?

Usage:
    python scripts/diagnose_jagged.py forecasts/a7004148bd5a4a2fae6062725d8dc090/
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast_dir")
    parser.add_argument("--var", default="u_500")
    args = parser.parse_args()

    pred_ds = xr.open_zarr(os.path.join(args.forecast_dir, "predictions.zarr"))
    truth_ds = xr.open_zarr(os.path.join(args.forecast_dir, "input_timeseries.zarr"))

    var = args.var
    pred = pred_ds[var].values  # (time, node)
    truth = truth_ds[var].values

    n = min(len(pred), len(truth))
    pred = pred[:n]
    truth = truth[:n]

    # 1. RMSE per step
    rmse = np.sqrt(((pred - truth) ** 2).mean(axis=1))
    print(f"RMSE per step for {var}:")
    for t, r in enumerate(rmse):
        diff = r - rmse[t - 1] if t > 0 else 0
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else " "
        print(f"  step {t:3d}: {r:.4f}  {arrow} {diff:+.4f}")

    # 2. Check if shifting by ±1 reduces RMSE (alignment test)
    if n > 2:
        rmse_shift_plus1 = np.sqrt(((pred[1:] - truth[:-1]) ** 2).mean(axis=1)).mean()
        rmse_shift_minus1 = np.sqrt(((pred[:-1] - truth[1:]) ** 2).mean(axis=1)).mean()
        rmse_aligned = np.sqrt(((pred - truth) ** 2).mean(axis=1)).mean()
        print(f"\nAlignment test (mean RMSE):")
        print(f"  Aligned (pred[t] vs truth[t]):     {rmse_aligned:.4f}")
        print(f"  Shift +1 (pred[t+1] vs truth[t]):  {rmse_shift_plus1:.4f}")
        print(f"  Shift -1 (pred[t] vs truth[t+1]):  {rmse_shift_minus1:.4f}")
        if rmse_shift_plus1 < rmse_aligned * 0.9 or rmse_shift_minus1 < rmse_aligned * 0.9:
            print("  ⚠ A shifted alignment has >10% lower RMSE — possible off-by-one!")
        else:
            print("  ✓ Current alignment looks correct")

    # 3. Plot RMSE per step
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    ax = axes[0]
    ax.plot(rmse, "o-", markersize=3)
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{var} RMSE per step")
    ax.grid(True, alpha=0.3)

    # 4. Step-to-step RMSE changes
    ax = axes[1]
    diffs = np.diff(rmse)
    colors = ["green" if d < 0 else "red" for d in diffs]
    ax.bar(range(len(diffs)), diffs, color=colors, alpha=0.7)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("ΔRMSE (step-to-step)")
    ax.set_title(f"{var} RMSE change per step (green=decrease, red=increase)")
    ax.grid(True, alpha=0.3)

    # 5. Single node timeseries
    ax = axes[2]
    node = 0
    ax.plot(truth[:, node], label="truth", linewidth=1.5)
    ax.plot(pred[:, node], label="pred", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel(var)
    ax.set_title(f"{var} at node 0: pred vs truth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(args.forecast_dir, f"diagnose_{var}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()

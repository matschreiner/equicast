"""Compare before_cons and after_cons forecast zarrs for a given run.

Usage:
    python scripts/compare_forecasts.py <run_name>
    python scripts/compare_forecasts.py abrasive-seal-529
"""

import sys
import numpy as np
import xarray as xr

RUN = sys.argv[1] if len(sys.argv) > 1 else "abrasive-seal-529"
BASE = f"forecasts/{RUN}"

before = xr.open_zarr(f"{BASE}/before_cons/forecast.zarr")
after = xr.open_zarr(f"{BASE}/after_cons/forecast.zarr")

variables = sorted(before.data_vars)
assert set(variables) == set(after.data_vars), "Variable mismatch between datasets"

print(f"Comparing {RUN}: before_cons vs after_cons")
print(f"  Shape: {dict(before.dims)}")
print(f"  Variables: {len(variables)}\n")

print(f"{'variable':<12}  {'max_abs_diff':>14}  {'mean_abs_diff':>14}  {'rmse':>10}  {'allclose':>8}")
print("-" * 65)

any_diff = False
for var in variables:
    b = before[var].values
    a = after[var].values
    diff = a - b
    max_diff = np.abs(diff).max()
    mean_diff = np.abs(diff).mean()
    rmse = np.sqrt((diff ** 2).mean())
    close = np.allclose(b, a, rtol=1e-4, atol=1e-6)
    if not close:
        any_diff = True
    print(f"{var:<12}  {max_diff:>14.6g}  {mean_diff:>14.6g}  {rmse:>10.6g}  {'YES' if close else 'NO':>8}")

print()
if any_diff:
    print("RESULT: datasets differ")
else:
    print("RESULT: datasets match (allclose within rtol=1e-4, atol=1e-6)")

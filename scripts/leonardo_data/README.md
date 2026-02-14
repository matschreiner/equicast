# Leonardo Data Scripts

Scripts for extracting and preparing timeseries data from the Leonardo HPC dataset for local forecasting.

## Workflow

1. **`save_timeseries.py`** — Run on Leonardo. Reads frames from the full zarr dataset and saves them as a compressed numpy archive (`timeseries_raw.npz`).

2. **`unpack_timeseries.py`** — Run locally. Loads the raw numpy archive, attaches each frame to a graph, and saves the result as a PyTorch timeseries (`timeseries.pt`).

The output `timeseries.pt` can then be used with `config/forecast_config.py` to run forecasts from a checkpoint.

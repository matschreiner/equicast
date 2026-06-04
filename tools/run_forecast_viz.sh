#!/bin/bash
# Usage: bash tools/run_forecast_viz.sh <forecast.json or forecast_dir>
set -e

FORECAST=${1:?Usage: run_forecast_viz.sh <forecast.json or forecast_dir>}

python tools/rmse_forecast.py "$FORECAST"
python tools/visualize_forecast.py "$FORECAST" wind_850
python tools/visualize_forecast.py "$FORECAST" z_500
python tools/visualize_forecast.py "$FORECAST" t_850
python tools/visualize_forecast.py "$FORECAST" q_700

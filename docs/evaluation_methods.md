# Evaluation Methods for ML Weather Models

Summary of how GraphCast, Pangu-Weather, and FuXi evaluate forecast skill, and the WeatherBench2 standard.

## Common Setup

All three models share:

| Aspect | Value |
|--------|-------|
| Ground truth | ERA5 reanalysis |
| Grid resolution | 0.25° lat/lon (721 x 1440) |
| Training data | ERA5 1979-2017 |
| Test year | 2018 (original papers), 2020 (WeatherBench2) |
| Forecast method | Autoregressive rollout, 6h steps |
| Init times | 00Z and 12Z daily (~730 per year) |
| Variables | 69: 5 upper-air x 13 pressure levels + 4 surface |

### Variables

**Upper-air** at 13 levels (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa):
- Geopotential (Z), Temperature (T), U-wind, V-wind, Specific humidity (Q)

**Surface** (4 vars):
- 2m temperature (T2m), 10m U-wind (U10), 10m V-wind (V10), Mean sea level pressure (MSLP)

**Headline variables** typically reported: Z500, T850, T2m, Q700, 10m wind speed.

## Metrics

### Latitude-Weighted RMSE

All three use the same formula (different notations, mathematically equivalent):

```
RMSE(v, τ) = √[ Σᵢ wᵢ (predᵢ - targetᵢ)² / Σᵢ wᵢ ]
```

where `wᵢ = cos(latᵢ)` corrects for grid cell area shrinking toward the poles.

Pangu-Weather writes it slightly differently:
```
L_i = (N_lat * cos(φᵢ)) / Σ cos(φₖ)
RMSE = √[ Σᵢⱼ Lᵢ (predᵢⱼ - targetᵢⱼ)² / (N_lat * N_lon) ]
```
This is equivalent — the `N_lat` in `Lᵢ` cancels with the `N_lat` in the denominator.

### Latitude-Weighted ACC (Anomaly Correlation Coefficient)

```
ACC(v, τ) = Σᵢ wᵢ · a_pred_i · a_target_i / √[ Σᵢ wᵢ · a_pred_i² · Σᵢ wᵢ · a_target_i² ]
```

where:
- `a_pred = pred - climatology`
- `a_target = target - climatology`
- Climatology = ERA5 long-term mean (daily or hourly, over ~1990-2017)

ACC measures correlation between predicted and observed anomalies. A value of 1.0 is perfect; the conventional "useful forecast" threshold is ACC > 0.6.

### Other Metrics

- **MAE** (mean absolute error): reported by GraphCast
- **Bias** (mean error): reported in WeatherBench2
- **CRPS**: only for probabilistic/ensemble models (FuXi-ENS, GenCast), not for deterministic models

## Per-Model Details

### GraphCast (Lam et al. 2023, Science)
- Lead times: 6h to **10 days** (240h), 6h intervals
- First major model to use **WeatherBench2** for evaluation
- Operates natively on 0.25° grid (no regridding needed)
- Predicts 227 variables (5 upper-air x 37 levels + 6 surface), reported on 13 standard levels
- HRES comparison: HRES regridded from ~0.1° to 0.25° for fair comparison

### Pangu-Weather (Bi et al. 2023, Nature)
- Lead times: up to **7 days**, 6h/24h steps
- Uses 4 separate models (1h, 3h, 6h, 24h step) combined hierarchically
- Does NOT use WeatherBench — own evaluation protocol
- 69 variables, 0.25° grid

### FuXi (Chen et al. 2023, npj Clim Atmos Sci)
- Lead times: up to **15 days** (360h), 6h intervals
- Cascade of 3 models: FuXi-Short (0-5d), FuXi-Medium (5-10d), FuXi-Long (10-15d)
- Does NOT use WeatherBench — own evaluation protocol
- 69 variables, 0.25° grid

## WeatherBench2 Standard

WeatherBench2 (Rasp et al. 2024, JAMES) is the emerging **de facto standard** for comparing ML weather models.

| Aspect | WeatherBench2 | Original papers |
|--------|--------------|-----------------|
| Test year | 2020 | 2018 |
| Eval grids | 1.5° and 0.25° | 0.25° (native) |
| Climatology | Standardized ERA5 hourly | Paper-specific |
| Regridding | Conservative (area-weighted) | Varies |
| Metrics code | Open-source, reproducible | Custom |
| Leaderboard | Public | N/A |

WB2 regrids all models to a **common 1.5° grid** for headline scorecard comparisons, ensuring fair comparison across different native resolutions.

Models evaluated on WB2: GraphCast, Pangu-Weather, FuXi, FourCastNet, ECMWF IFS HRES/ENS, climatology, persistence.

## WeatherBenchX Data Format

WeatherBenchX (the evaluation library) expects:

```
Dimensions:  (init_time, lead_time, latitude, longitude)
Coordinates:
  * init_time   datetime64[ns]
  * lead_time   timedelta64[ns]
  * latitude    float64          (degrees, monotonic)
  * longitude   float64          (degrees, monotonic)
Data variables:
    <var_name>  (init_time, lead_time, latitude, longitude) float32
```

- Requires **regular lat/lon grid** (no unstructured/node grids)
- Dimension names must be `latitude`/`longitude` (not `lat`/`lon`)
- Latitude weighting handled by `GridAreaWeighting` class
- Targets can use `valid_time` instead of `init_time + lead_time`

## Implications for Equicast

To compare with these models, the forecaster must:
1. **Regrid** from unstructured graph nodes to regular 0.25° lat/lon grid
2. Save output with dims `(init_time, lead_time, latitude, longitude)` and proper dtypes
3. Run forecasts from **multiple init times** across the test period
4. Use **ERA5** as ground truth
5. Compute latitude-weighted RMSE and ACC via WeatherBenchX

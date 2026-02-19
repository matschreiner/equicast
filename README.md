# Equicast

Equivariant graph neural network for weather forecasting on ERA5 data.

## Overview

Equicast trains GNN-based weather models on ERA5 reanalysis data using an encoder–processor–decoder architecture over an icosahedral mesh graph (GraphCast-style). Two model families are supported:

- **PaiNN** — SE(3)-equivariant model trained with an equivariant MSE loss
- **DiffusionPaiNN** — DDPM model using a cosine noise schedule

Data handling uses [anemoi-datasets](https://github.com/ecmwf/anemoi-datasets) and [anemoi-graphs](https://github.com/ecmwf/anemoi-graphs). Experiments are configured with [Fiddle](https://github.com/google/fiddle) and tracked with MLFlow.

## Installation

```bash
pip install -e .
```

Dependencies (installed automatically): `mlflow`, `anemoi-datasets`, `anemoi-graphs`, `anemoi-models`, `pytorch_lightning`, `torch-geometric`, `trimesh`, `cartopy`, `fiddle`, `tqdm`.

## Usage

### Train deterministic model (PaiNN)

```bash
python config/train.py
```

### Train diffusion model (DiffusionPaiNN)

```bash
python config/diffusion_train.py
```

To customise hyperparameters, pass a fiddler — a Python snippet that mutates the Fiddle config:

```bash
python config/train.py --fiddler config/fiddlers/leonardo.py
```

### Run forecast

```bash
python config/forecast.py <run_name>
```

`<run_name>` is the MLFlow run name of the trained model. Forecasts are saved to `forecasts/`.

## Project Structure

```
config/          Training and forecast entry points + fiddlers
equicast/
  data/          Dataset, graph attachment, normalization, data handlers
  model/         Model Lightning module, backbones (PaiNN, DiffusionPaiNN), diffusion utilities
  experiment/    TrainConfig, ForecastConfig, run_experiment
  forecaster.py  Autoregressive rollout
  logger/        MLFlow logger wrapper
  metrics.py     Latitude-weighted RMSE and ACC
  validation/    Validation loop utilities
graph/           Pre-built icosahedral mesh graphs (.pt)
docs/            Notes on evaluation methods, architecture decisions
```

## Evaluation

Forecasts are evaluated against ERA5 using latitude-weighted RMSE and ACC (Anomaly Correlation Coefficient), following the [WeatherBench2](https://github.com/google-deepmind/weatherbench2) standard. Key headline variables: Z500, T850, T2m, U10/V10.

## Author

Mathias Schreiner

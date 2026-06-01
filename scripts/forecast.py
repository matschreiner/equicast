"""Autoregressive forecast from a trained model checkpoint."""

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from equicast.experiment.forecast import ForecastConfig
from equicast.forecaster import Forecaster
from equicast.logger import MLFlowLogger
from equicast.model.base import BaseModel
from equicast.model.from_checkpoint import load_from_checkpoint


def build_forecast_config(cfg: DictConfig) -> ForecastConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_from_checkpoint(BaseModel, cfg.ckpt_path)
    model.eval()
    model.to(device)
    data_handler = model.data_handler

    variables = [v for v, _ in sorted(data_handler.name_to_index.items(), key=lambda x: x[1])]
    graph_provider = instantiate(cfg.data.graph_provider)
    dataset = instantiate(cfg.data.dataset, path=cfg.data.dataset_path, graph_provider=graph_provider, no_frames=1, variables=variables)

    start_idx = cfg.forecast.get("start_idx", 0)
    n_steps = cfg.forecast.n_steps
    n_input = getattr(data_handler, "n_input_frames", 1)
    timeseries = [dataset[start_idx + i][0] for i in range(n_input + n_steps)]

    experiment_dir = cfg.ckpt_path.split("/mlflow/")[0] if "/mlflow/" in cfg.ckpt_path else "."
    run_id = cfg.ckpt_path.split("/artifacts/")[0].rsplit("/", 1)[-1] if "/artifacts/" in cfg.ckpt_path else "forecast"

    return ForecastConfig(
        forecaster=Forecaster(model=model),
        input_timeseries=timeseries,
        data_handler=data_handler,
        model_id=run_id,
        output_dir=f"{experiment_dir}/forecasts",
        meta={
            "model_id": run_id,
            "ckpt_path": cfg.ckpt_path,
            "dataset": cfg.data.dataset_path,
            "start_idx": start_idx,
            "n_steps": n_steps,
        },
    )


@hydra.main(config_path="../config", config_name="forecast", version_base=None)
def main(cfg: DictConfig):
    assert cfg.ckpt_path, "ckpt_path must be set"
    forecast_config = build_forecast_config(cfg)
    forecast_config.run()


if __name__ == "__main__":
    main()

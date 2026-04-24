"""Autoregressive forecast from a trained model checkpoint."""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from equicast import TRACKING_URI
from equicast.experiment.forecast import ForecastConfig
from equicast.forecaster import Forecaster
from equicast.logger import MLFlowLogger
from equicast.model.base import BaseModel
from equicast.model.from_checkpoint import load_from_checkpoint


def build_forecast_config(cfg: DictConfig) -> ForecastConfig:
    feature_config = instantiate(cfg.data.feature_config)
    graph_provider = instantiate(cfg.data.graph_provider)
    data_handler = instantiate(cfg.data.handler, dataset_path=cfg.data.dataset_path, feature_config=feature_config)
    dataset = instantiate(cfg.data.dataset, path=cfg.data.dataset_path, graph_provider=graph_provider, no_frames=1)

    model = load_from_checkpoint(BaseModel, cfg.ckpt_path, data_handler=data_handler, weights_only=False)
    model.eval()

    logger = MLFlowLogger(experiment_name=cfg.logger.experiment_name, tracking_uri=TRACKING_URI)

    forecast_cfg = cfg.forecast
    start_idx = forecast_cfg.get("start_idx", 0)
    n_steps = forecast_cfg.n_steps
    timeseries = [dataset[start_idx + i][0] for i in range(n_steps + 1)]

    return ForecastConfig(
        forecaster=Forecaster(model=model, logger=logger),
        input_timeseries=timeseries,
        target_timeseries=timeseries[1:],
        logger=logger,
        data_handler=data_handler,
        model_id=logger.run_id or "forecast",
    )


@hydra.main(config_path="conf", config_name="forecast", version_base=None)
def main(cfg: DictConfig):
    assert cfg.ckpt_path, "ckpt_path must be set"
    forecast_config = build_forecast_config(cfg)
    forecast_config.run()


if __name__ == "__main__":
    main()

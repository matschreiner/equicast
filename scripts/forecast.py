"""Autoregressive forecast from a trained model checkpoint."""

import json
import os

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from equicast.experiment.forecast import ForecastConfig
from equicast.forecaster import Forecaster
from equicast.model.base import BaseModel
from equicast.model.from_checkpoint import load_from_checkpoint


@hydra.main(config_path="../config", config_name="forecast", version_base=None)
def main(cfg: DictConfig):
    assert cfg.ckpt_path, "ckpt_path must be set"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_from_checkpoint(BaseModel, cfg.ckpt_path)
    model.eval()
    model.to(device)
    data_handler = model.data_handler

    variables = [v for v, _ in sorted(data_handler.name_to_index.items(), key=lambda x: x[1])]
    graph_provider = instantiate(cfg.data.graph_provider)
    dataset = instantiate(cfg.data.dataset, path=cfg.data.dataset_path, graph_provider=graph_provider, no_frames=1, variables=variables)

    n_steps = cfg.forecast.n_steps
    n_input = getattr(data_handler, "n_input_frames", 1)
    start_idx = cfg.forecast.get("start_idx", 0)
    n_starts = cfg.forecast.get("n_starts", 1)
    start_every = cfg.forecast.get("start_every", 1)
    output_name = cfg.forecast.get("output_name", "")
    batch = n_starts > 1

    experiment_dir = os.path.abspath(cfg.ckpt_path.split("/mlflow/")[0]) if "/mlflow/" in cfg.ckpt_path else os.getcwd()
    run_id = cfg.ckpt_path.split("/artifacts/")[0].rsplit("/", 1)[-1] if "/artifacts/" in cfg.ckpt_path else "forecast"

    if batch:
        # Save global eval metadata once
        from datetime import datetime
        name = output_name or datetime.now().strftime("%Y-%m-%d_%H:%M")
        output_dir = f"{experiment_dir}/forecasts/{run_id}/{name}"
        os.makedirs(output_dir, exist_ok=True)
        eval_meta = {
            "model_id": run_id,
            "ckpt_path": cfg.ckpt_path,
            "dataset": cfg.data.dataset_path,
            "n_steps": n_steps,
            "start_idx": start_idx,
            "n_starts": n_starts,
            "start_every": start_every,
            "output_dir": output_dir,
        }
        with open(os.path.join(output_dir, "eval.json"), "w") as f:
            json.dump(eval_meta, f, indent=2)

    for i in range(n_starts):
        current_start = start_idx + i * start_every
        timeseries = [dataset[current_start + j][0] for j in range(n_input + n_steps)]
        ForecastConfig(
            forecaster=Forecaster(model=model),
            input_timeseries=timeseries,
            data_handler=data_handler,
            model_id=run_id,
            output_dir=f"{experiment_dir}/forecasts",
            output_name=output_name,
            meta={
                "model_id": run_id,
                "ckpt_path": cfg.ckpt_path,
                "dataset": cfg.data.dataset_path,
                "start_idx": current_start,
                "n_steps": n_steps,
                "n_input": n_input,
                "batch": batch,
            },
        ).run()
        if batch:
            print(f"[{i+1}/{n_starts}] start_idx={current_start}")


if __name__ == "__main__":
    main()

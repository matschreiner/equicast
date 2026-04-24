"""Sample from a trained DiffusionModel given a conditioning state from the dataset."""

import json
from datetime import datetime
from pathlib import Path

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from equicast.model.base import BaseModel
from equicast.model.from_checkpoint import load_from_checkpoint


@hydra.main(config_path="conf", config_name="sample", version_base=None)
def main(cfg: DictConfig):
    assert cfg.ckpt_path, "ckpt_path must be set"

    feature_config = instantiate(cfg.data.feature_config)
    graph_provider = instantiate(cfg.data.graph_provider)
    data_handler = instantiate(cfg.data.handler, dataset_path=cfg.data.dataset_path, feature_config=feature_config)
    dataset = instantiate(cfg.data.dataset, path=cfg.data.dataset_path, graph_provider=graph_provider, no_frames=2)

    model = load_from_checkpoint(BaseModel, cfg.ckpt_path, data_handler=data_handler, weights_only=False)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    run_id = Path(cfg.ckpt_path).parent.name
    timestamp = datetime.now().strftime("%y.%m.%d_%H:%M:%S")
    out_dir = Path(cfg.sample.out_path or f"samples/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({"run_id": run_id, "ckpt_path": cfg.ckpt_path}, indent=2))

    frame_idx = cfg.sample.frame_idx
    physical_input, physical_target = dataset[frame_idx]
    physical_input = physical_input.to(device)
    save_trajectory = cfg.sample.save_trajectory

    with torch.no_grad():
        result = model.predict(physical_input, return_trajectory=save_trajectory)

    if save_trajectory:
        physical_output, trajectory = result
    else:
        physical_output = result

    nodes = data_handler.nodes
    physical_out_scalars = data_handler.get_output_scalars(physical_output[nodes].data)
    ground_truth_scalars = data_handler.get_output_scalars(physical_target[nodes].data)

    torch.save(physical_out_scalars.cpu(), out_dir / "sample.pt")
    torch.save(ground_truth_scalars.cpu(), out_dir / "ground_truth.pt")
    print(f"Saved sample [{physical_out_scalars.shape}] → {out_dir}/sample.pt")

    if save_trajectory:
        torch.save(trajectory.cpu(), out_dir / "trajectory.pt")
        print(f"Saved trajectory [{trajectory.shape}] → {out_dir}/trajectory.pt")


if __name__ == "__main__":
    main()

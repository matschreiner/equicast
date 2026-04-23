"""Sample from a trained DiffusionModel given a conditioning state from the dataset."""

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from equicast.model.diffusion import DiffusionModel


def build_sample_config(cfg: DictConfig):
    feature_config = instantiate(cfg.data.feature_config)
    graph_provider = instantiate(cfg.data.graph_provider)
    data_handler = instantiate(cfg.data.handler, dataset_path=cfg.data.dataset_path, feature_config=feature_config)
    dataset = instantiate(cfg.data.dataset, path=cfg.data.dataset_path, graph_provider=graph_provider)

    model = DiffusionModel.load_from_checkpoint(
        cfg.ckpt_path,
        data_handler=data_handler,
        weights_only=False,
    )
    model.eval()

    return model, data_handler, dataset


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    assert cfg.ckpt_path, "ckpt_path must be set"

    model, data_handler, dataset = build_sample_config(cfg)

    frame_idx = cfg.get("frame_idx", 0)
    batch = dataset[frame_idx]
    input_ = data_handler.prepare_backbone_input(batch[0])

    with torch.no_grad():
        sample = model.predict(input_)

    sample_denorm = data_handler.normalizer.denormalize_output(sample)

    out_path = cfg.get("out_path", "sample.pt")
    torch.save(sample_denorm.cpu(), out_path)
    print(f"Saved sample [{sample_denorm.shape}] → {out_path}")


if __name__ == "__main__":
    main()

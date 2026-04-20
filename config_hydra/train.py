import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    instantiate(cfg).run()


if __name__ == "__main__":
    main()

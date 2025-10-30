from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

#  CONFIG_PATH = Path(__file__).resolve().parents[1] / "config"
CONFIG_PATH = "/home/masc/dmi/programming/equicast/config"


@hydra.main(config_path=str(CONFIG_PATH), config_name="config.yaml", version_base=None)
def main(cfg: DictConfig):
    variables = cfg.dataset.variables
    model = instantiate(cfg.model, variables=variables)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    trainer = instantiate(
        cfg.trainer,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    dataset = instantiate(cfg.dataset)
    trainer.fit(model, dataset)


#

if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf


def main(cfg):
    dataset = instantiate(cfg.dataset)
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    logger = instantiate(cfg.logger)
    logger.log_hyperparams({"config": cfg})

    trainer = instantiate(
        cfg.trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
    )

    trainer.fit(model, dataset)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("config_path")

    args = argparser.parse_args()
    cfg = OmegaConf.load(args.config_path)

    OmegaConf.resolve(cfg)
    main(cfg)

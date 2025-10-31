import argparse

from hydra.utils import instantiate
from omegaconf import OmegaConf


def main(cfg):
    dataset = instantiate(cfg.dataset)
    instantiate(cfg.model)
    variables = cfg.variables
    model = instantiate(cfg.model, variables=variables)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    logger = instantiate(cfg.logger)

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

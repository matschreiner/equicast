import argparse

from hydra.utils import instantiate
from omegaconf import OmegaConf


def main(cfg):
    #  def t(batch):
    #      __import__("pdb").set_trace()  # TODO delme
    #      print(batch)
    #      return batch
    #
    dataset = instantiate(cfg.dataset, transforms=[t])
    variables = cfg.dataset.variables
    model = instantiate(cfg.model, variables=variables)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)
    logger = instantiate(cfg.logger)
    graph_provider = instantiate(cfg.graph_provider)

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

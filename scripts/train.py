from hydra.utils import instantiate

import equicast

def main(cfg):
    # Data
    dataset = instantiate(cfg.dataset)
    #  scaler = instantiate(cfg.scaler, statistics=dataset.statistics)
    #  router = instantiate(cfg.feature_router, name_to_idx=dataset.data.name_to_index)

    scaler = equicast.data.scaler Scaler(dataset.statistics)
    __import__("pdb").set_trace() #TODO delme 

    # Model
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # Logger
    logger = instantiate(cfg.logger)
    logger.log_hyperparams({"config": cfg})

    trainer = instantiate(
        cfg.trainer,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
    )

    trainer.fit(model, dataset)


import sys

from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig

if __name__ == "__main__":
    overrides = sys.argv[1:]

    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    main(cfg)

#  if __name__ == "__main__":
#      argparser = argparse.ArgumentParser()
#      argparser.add_argument("config_path")
#
#      args = argparser.parse_args()
#      cfg = OmegaConf.load(args.config_path)
#
#      OmegaConf.resolve(cfg)
#      main(cfg)

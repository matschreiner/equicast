import sys

from hydra import compose, initialize
from hydra.utils import instantiate

from equicast.data.feature_router import FeatureRouter
from equicast.data.scaler import Scaler


def main(cfg):
    dataset = instantiate(cfg.dataset)

    scaler = Scaler(statistics=dataset.statistics)
    feature_router = FeatureRouter(
        features=cfg.features,
        name_to_index=dataset.name_to_index,
    )
    backbone = instantiate(
        cfg.backbone,
        in_dim=feature_router.in_dim,
        out_dim=feature_router.out_dim,
    )

    model = instantiate(
        cfg.model,
        backbone=backbone,
        scaler=scaler,
        feature_router=feature_router,
    )

    dataloader = instantiate(cfg.dataloader, dataset=dataset)
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
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    overrides = sys.argv[1:]

    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name="config", overrides=overrides)

    main(cfg)

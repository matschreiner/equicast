import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from equicast.experiment.train import TrainConfig
from equicast.logger import MLFlowLogger
from equicast.metrics import WeatherBenchTracker


def build_train_config(cfg: DictConfig) -> TrainConfig:
    feature_config = instantiate(cfg.data.feature_config)
    graph_provider = instantiate(cfg.data.graph_provider)
    data_handler = instantiate(cfg.model.data_handler, dataset_path=cfg.data.dataset_path, feature_config=feature_config)
    dataset = instantiate(cfg.data.dataset, path=cfg.data.dataset_path, graph_provider=graph_provider)
    dataloader = instantiate(cfg.data.dataloader, dataset=dataset)

    backbone = instantiate(cfg.model.backbone, data_handler=data_handler)

    loss_cfg = cfg.model.get("loss")
    optimizer_factory = instantiate(cfg.optimization.optimizer)

    scheduler_cfg = cfg.optimization.get("scheduler")
    scheduler_factory = instantiate(scheduler_cfg) if scheduler_cfg else None
    model_kwargs = dict(
        backbone=backbone,
        data_handler=data_handler,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        metrics_tracker=WeatherBenchTracker(data_handler=data_handler),
    )
    if loss_cfg:
        model_kwargs["loss_fn"] = instantiate(loss_cfg)
    model = instantiate(cfg.model.model, **model_kwargs)

    logger = MLFlowLogger(experiment_name=cfg.logger.experiment_name, tracking_uri=cfg.logger.tracking_uri)
    logger.log_config(cfg)
    callbacks = [instantiate(cb) for cb in cfg.trainer.callbacks]
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    return TrainConfig(
        model=model,
        trainer=trainer,
        dataloader=dataloader,
        logger=logger,
        ckpt_path=cfg.get("ckpt_path"),
    )


@hydra.main(config_path="../config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    train_config = build_train_config(cfg)
    train_config.run()


if __name__ == "__main__":
    main()

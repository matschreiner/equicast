import fiddle as fdl

from equicast.logger import CSVLogger


def fiddler(cfg: fdl.Config) -> None:
    leonardo_dataset = "/leonardo_work/DE360_drusso/aifs/aifs.zarr"
    cfg.model.backbone.data_handler.dataset_path = leonardo_dataset
    cfg.dataloader.dataset.path = leonardo_dataset
    cfg.dataloader.num_workers = 8
    cfg.dataloader.pin_memory = True
    cfg.trainer.devices = 4
    cfg.trainer.strategy = "ddp"

    logger = fdl.Config(
        CSVLogger,
        save_dir="logs",
        name="equicast",
    )
    cfg.logger = logger
    cfg.trainer.logger = logger

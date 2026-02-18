import fiddle as fdl


def fiddler(cfg: fdl.Config) -> None:
    leonardo_era5 = (
        "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"
    )

    cfg.model.backbone.data_handler.dataset_path = leonardo_era5
    cfg.dataloader.dataset.path = leonardo_era5
    cfg.dataloader.dataset.step = 6
    cfg.dataloader.num_workers = 7
    cfg.dataloader.batch_size = 1
    cfg.trainer.strategy = "ddp"

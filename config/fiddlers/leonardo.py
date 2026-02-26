import fiddle as fdl

from config.fiddle_tags import DatasetPath

DATASET_PATH = "/leonardo_work/DestE_340_26/ai-ml/datasets/aifs-ea-an-oper-0001-mars-o96-1979-2024-1h-v3-with-era51.zarr"


def fiddler(cfg: fdl.Config) -> None:
    fdl.set_tagged(cfg, tag=DatasetPath, value=DATASET_PATH)
    cfg.dataloader.dataset.step = 6
    cfg.dataloader.num_workers = 7
    cfg.dataloader.batch_size = 1
    cfg.trainer.strategy = "ddp"

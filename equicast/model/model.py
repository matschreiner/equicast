import warnings
from contextlib import contextmanager

import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(
        self,
        backbone,
        processor,
    ):
        super().__init__()
        with ignore_backbone_warning():
            self.save_hyperparameters()

        self.backbone = backbone
        self.processor = processor

    def forward(self, batch):
        batch = self.backbone(batch)
        return batch

    def training_step(self, batch, batch_idx):
        batch["input"] = processor.prepare(batch["input"])
        out = batch["target"]

        batch = self.processor.preprocess(batch)
        batch = self.forward(batch)

        loss = ((pred - target) ** 2).mean()
        return loss


@contextmanager
def ignore_backbone_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Attribute 'backbone' is an instance of `nn\.Module`",
        )
        yield

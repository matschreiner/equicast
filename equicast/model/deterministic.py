import torch
import torch.nn as nn

from equicast.model.base import BaseModel, default_optimizer_factory, ignore_module_warning
from equicast.model.losses import EquivariantMSELoss, MSELoss

# Alias for backwards compatibility with old checkpoints
equivariant_loss_fn = EquivariantMSELoss


class Deterministic(BaseModel):
    def __init__(
        self,
        backbone: nn.Module,
        data_handler,
        loss_fn: nn.Module = None,
        optimizer_factory=default_optimizer_factory,
        scheduler_factory=None,
        metrics_tracker=None,
        compile_backbone: bool = True,
    ):
        super().__init__(
            data_handler=data_handler,
            optimizer_factory=optimizer_factory,
            scheduler_factory=scheduler_factory,
            metrics_tracker=metrics_tracker,
        )
        with ignore_module_warning("backbone", "data_handler"):
            self.save_hyperparameters(ignore=["loss_fn"])

        self.backbone = torch.compile(backbone) if compile_backbone else backbone
        self.loss_fn = MSELoss() if loss_fn is None else loss_fn

    def predict(self, physical_input):
        physical_input = physical_input.clone()
        backbone_input = self.data_handler.prepare_backbone_input(physical_input)
        backbone_out = self.backbone(backbone_input)
        return self.data_handler.update_state_with_backbone_output(physical_input, backbone_out)

    def training_step(self, batch, _):
        backbone_input, backbone_target = self.data_handler.prepare_training_batch(batch)
        backbone_out = self.backbone(backbone_input)
        loss = self.loss_fn(backbone_out, backbone_target)

        if self._should_log_metrics():
            self.log_lr()
            self.log_loss(loss, backbone_input.num_graphs)
            self.log_metrics(backbone_out, backbone_target)

        return loss

    def validation_step(self, batch, _):
        backbone_input, backbone_target = self.data_handler.prepare_training_batch(batch)
        backbone_out = self.backbone(backbone_input)
        loss = self.loss_fn(backbone_out, backbone_target)

        self.log(
            "val/loss", loss, logger=True, prog_bar=False, on_step=False, on_epoch=True, batch_size=backbone_input.num_graphs
        )
        return loss

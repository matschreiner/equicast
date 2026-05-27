import torch
import torch.nn as nn

from equicast.model.base import BaseModel, default_optimizer_factory
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
        self.backbone = torch.compile(backbone) if compile_backbone else backbone
        self.loss_fn = MSELoss() if loss_fn is None else loss_fn

    def predict(self, phys_input):
        backbone_input = self.data_handler.prepare_backbone_input(phys_input)
        backbone_output = self.backbone(backbone_input)
        return self.data_handler.update_phys_with_backbone_output(backbone_input, backbone_output)

    def training_step(self, batch, _):
        backbone_input, backbone_target = self.data_handler.prepare_training_batch(batch)
        backbone_output = self.backbone(backbone_input)
        loss = self.loss_fn(backbone_output, backbone_target)

        if self._should_log_metrics():
            self.log_lr()
            self.log_loss(loss, backbone_input.num_graphs)
            self.log_metrics(backbone_output, backbone_target)

        return loss

    def rollout_training_step(self, batch, n_rollout_steps: int):
        """Multi-step autoregressive training. batch must have n_input_frames + n_rollout_steps frames."""
        n_input = getattr(self.data_handler, "n_input_frames", 1)
        window = batch[:n_input] if n_input > 1 else batch[0]

        total_loss = 0.0
        for i in range(n_rollout_steps):
            phys_target = batch[n_input + i]
            backbone_input = self.data_handler.prepare_backbone_input(window)
            backbone_output = self.backbone(backbone_input)
            backbone_target = self.data_handler.prepare_backbone_target(phys_target)
            total_loss = total_loss + self.loss_fn(backbone_output, backbone_target)

            next_state = self.data_handler.update_phys_with_backbone_output(phys_target, backbone_output)

            window = window[1:] + [next_state] if n_input > 1 else next_state

        return total_loss / n_rollout_steps

    def validation_step(self, batch, _):
        backbone_input, backbone_target = self.data_handler.prepare_training_batch(batch)
        backbone_output = self.backbone(backbone_input)
        loss = self.loss_fn(backbone_output, backbone_target)

        self.log(
            "val/loss",
            loss,
            logger=True,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=backbone_input.num_graphs,
        )
        return loss

import torch


class MSELoss(torch.nn.Module):
    def forward(self, backbone_out, backbone_target):
        return torch.nn.functional.mse_loss(backbone_out, backbone_target)


class EquivariantMSELoss(torch.nn.Module):
    def forward(self, backbone_out, backbone_target):
        scalar_loss = torch.nn.functional.mse_loss(backbone_out["scalar"], backbone_target["scalar"])
        vector_loss = torch.nn.functional.mse_loss(backbone_out["vector"], backbone_target["vector"])
        return scalar_loss + vector_loss

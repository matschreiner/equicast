import torch


class MSELoss(torch.nn.Module):
    def forward(self, backbone_output, backbone_target):
        return torch.nn.functional.mse_loss(backbone_output, backbone_target)


class EquivariantMSELoss(torch.nn.Module):
    def forward(self, backbone_output, backbone_target):
        scalar_loss = torch.nn.functional.mse_loss(backbone_output["scalar"], backbone_target["scalar"])
        vector_loss = torch.nn.functional.mse_loss(backbone_output["vector"], backbone_target["vector"])
        return scalar_loss + vector_loss

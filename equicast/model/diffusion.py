"""EDM2-style conditional diffusion model (scalar features only).

Reference: Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based Generative Models"
           Karras et al. (2024) "Analyzing and Improving the Training Dynamics of Diffusion Models"
"""

import math

import torch
import torch.nn as nn
from tqdm import tqdm

from equicast.model.base import BaseModel, default_optimizer_factory, ignore_module_warning


class FourierEmbedding(nn.Module):
    """Random Fourier features for noise level conditioning (EDM2 eq. 75)."""

    def __init__(self, dim: int, bandwidth: float = 1.0):
        super().__init__()
        self.register_buffer("freqs", 2 * math.pi * torch.randn(dim) * bandwidth)
        self.register_buffer("phases", 2 * math.pi * torch.rand(dim))

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        c_noise = (sigma.log() / 4).float()  # [1]
        y = c_noise.outer(self.freqs) + self.phases  # [1, dim]
        return (y.cos() * math.sqrt(2)).to(sigma.dtype)


class DiffusionModel(BaseModel):
    """Scalar-only conditional diffusion model with EDM2 training dynamics.

    Backbone is conditioned on the current state and denoises scalar target features.
    Backbone must be initialized with in_dim = original_in_dim + out_dim + noise_emb_dim
    (use build_diffusion_painn).
    """

    def __init__(
        self,
        backbone: nn.Module,
        data_handler,
        sigma_data: float = 0.5,
        P_mean: float = -0.4,
        P_std: float = 1.0,
        noise_emb_dim: int = 64,
        num_sampler_steps: int = 32,
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
        self.sigma_data = sigma_data
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sampler_steps = num_sampler_steps
        self.noise_emb = FourierEmbedding(noise_emb_dim)
        self.logvar_fourier = FourierEmbedding(noise_emb_dim)
        self.logvar_linear = nn.Linear(noise_emb_dim, 1)

    # --- EDM2 preconditioning (Karras et al. 2022, Table 1) ---

    def _c_skip(self, sigma):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def _c_out(self, sigma):
        return sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()

    def _c_in(self, sigma):
        return 1.0 / (sigma**2 + self.sigma_data**2).sqrt()

    def _loss_weight(self, sigma):
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def _sample_sigma(self, device):
        return (torch.randn(1, device=device) * self.P_std + self.P_mean).exp()

    def _logvar(self, sigma):
        return self.logvar_linear(self.logvar_fourier(sigma)).squeeze(-1)  # [1]

    def _denoise(self, backbone_input, noisy_scalar: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """D(x, σ) = c_skip·x + c_out·F(conditioning | c_in·x | noise_emb)."""
        nodes = self.data_handler.nodes
        backbone_input = backbone_input.clone()

        n_nodes = backbone_input[nodes].input_scalar.shape[0]
        noise_emb = self.noise_emb(sigma).expand(n_nodes, -1)  # [nodes, noise_emb_dim]

        backbone_input[nodes].input_scalar = torch.cat(
            [backbone_input[nodes].input_scalar, self._c_in(sigma) * noisy_scalar, noise_emb], dim=-1
        )
        # Zero scalar residual — EDM2 skip connection is c_skip * noisy_scalar below
        backbone_input[nodes].residual_scalar = torch.zeros_like(noisy_scalar)

        F_x = self.backbone(backbone_input)["scalar"]
        return self._c_skip(sigma) * noisy_scalar + self._c_out(sigma) * F_x

    # --- Training ---

    def training_step(self, batch, _):
        backbone_input, backbone_target = self.data_handler.prepare_training_batch(batch)
        backbone_target = backbone_target["scalar"]
        sigma = self._sample_sigma(backbone_input[self.data_handler.nodes].input_scalar.device)

        noisy = backbone_target + torch.randn_like(backbone_target) * sigma
        D_x = self._denoise(backbone_input, noisy, sigma)

        weight = self._loss_weight(sigma)
        logvar = self._logvar(sigma)
        loss = ((weight / logvar.exp()) * (D_x - backbone_target).pow(2).mean() + logvar).mean()

        if self._should_log_metrics():
            self.log_lr()
            self.log_loss(loss, backbone_input.num_graphs)

        return loss

    def validation_step(self, batch, _):
        backbone_input, backbone_target = self.data_handler.prepare_training_batch(batch)
        backbone_target = backbone_target["scalar"]
        sigma = self._sample_sigma(backbone_input[self.data_handler.nodes].input_scalar.device)

        noisy = backbone_target + torch.randn_like(backbone_target) * sigma
        D_x = self._denoise(backbone_input, noisy, sigma)

        loss = (self._loss_weight(sigma) * (D_x - backbone_target).pow(2).mean()).mean()
        self.log("val/loss", loss, logger=True, prog_bar=False, on_step=False, on_epoch=True, batch_size=backbone_input.num_graphs)
        return loss

    # --- Inference: Heun ODE sampler (EDM Algorithm 1) ---

    def _sample(self, backbone_input, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0,
                return_trajectory: bool = False):
        """Run Heun ODE sampler on a prepared backbone input graph. Returns backbone-space scalars."""
        nodes = self.data_handler.nodes
        device = backbone_input[nodes].input_scalar.device
        n_nodes = backbone_input[nodes].input_scalar.shape[0]
        steps = self.num_sampler_steps

        idx = torch.arange(steps, device=device, dtype=torch.float32)
        sigmas = (sigma_max ** (1 / rho) + idx / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

        x = torch.randn(n_nodes, self.data_handler.out_dim, device=device) * sigma_max
        trajectory = [x.clone()] if return_trajectory else None

        for i in tqdm(range(steps), desc="Sampling"):
            s_cur, s_next = sigmas[i : i + 1], sigmas[i + 1 : i + 2]

            D_cur = self._denoise(backbone_input, x, s_cur)
            d_cur = (x - D_cur) / s_cur
            x_next = x + (s_next - s_cur) * d_cur

            if s_next.item() > 0:  # Heun 2nd-order correction
                D_next = self._denoise(backbone_input, x_next, s_next)
                d_next = (x_next - D_next) / s_next
                x = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
            else:
                x = x_next

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, torch.stack(trajectory)  # [steps+1, n_nodes, out_dim]
        return x

    def predict(self, phys_input, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0,
                return_trajectory: bool = False):
        phys_input = phys_input.clone()
        backbone_input = self.data_handler.prepare_backbone_input(phys_input)
        result = self._sample(backbone_input, sigma_min, sigma_max, rho, return_trajectory=return_trajectory)
        if return_trajectory:
            backbone_output, trajectory = result
            physical_out = self.data_handler.update_state_with_backbone_outputput(phys_input, backbone_output)
            return physical_out, trajectory
        return self.data_handler.update_state_with_backbone_outputput(phys_input, result)

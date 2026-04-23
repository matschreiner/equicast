"""EDM2-style conditional diffusion model (scalar features only).

Reference: Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based Generative Models"
           Karras et al. (2024) "Analyzing and Improving the Training Dynamics of Diffusion Models"
"""

import math

import torch
import torch.nn as nn

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
        with ignore_module_warning("backbone", "data_handler"):
            self.save_hyperparameters()

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

    def _denoise(self, input_graph, noisy_scalar: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """D(x, σ) = c_skip·x + c_out·F(conditioning | c_in·x | noise_emb)."""
        nodes = self.data_handler.nodes
        graph = input_graph.clone()

        n_nodes = graph[nodes].input_scalar.shape[0]
        noise_emb = self.noise_emb(sigma).expand(n_nodes, -1)  # [nodes, noise_emb_dim]

        graph[nodes].input_scalar = torch.cat(
            [graph[nodes].input_scalar, self._c_in(sigma) * noisy_scalar, noise_emb], dim=-1
        )
        # Zero scalar residual — EDM2 skip connection is c_skip * noisy_scalar below
        graph[nodes].residual_scalar = torch.zeros_like(noisy_scalar)

        F_x = self.backbone(graph)["scalar"]
        return self._c_skip(sigma) * noisy_scalar + self._c_out(sigma) * F_x

    # --- Training ---

    def training_step(self, batch, _):
        input_, target = self.data_handler.prepare_training_batch(batch)
        target = target["scalar"]
        sigma = self._sample_sigma(input_[self.data_handler.nodes].input_scalar.device)

        noisy = target + torch.randn_like(target) * sigma
        D_x = self._denoise(input_, noisy, sigma)

        weight = self._loss_weight(sigma)
        logvar = self._logvar(sigma)
        loss = ((weight / logvar.exp()) * (D_x - target).pow(2).mean() + logvar).mean()

        if self._should_log_metrics():
            self.log_lr()
            self.log_loss(loss, input_.num_graphs)

        return loss

    def validation_step(self, batch, _):
        input_, target = self.data_handler.prepare_training_batch(batch)
        target = target["scalar"]
        sigma = self._sample_sigma(input_[self.data_handler.nodes].input_scalar.device)

        noisy = target + torch.randn_like(target) * sigma
        D_x = self._denoise(input_, noisy, sigma)

        loss = (self._loss_weight(sigma) * (D_x - target).pow(2).mean()).mean()
        self.log("val/loss", loss, logger=True, prog_bar=False, on_step=False, on_epoch=True, batch_size=input_.num_graphs)
        return loss

    # --- Inference: Heun ODE sampler (EDM Algorithm 1) ---

    def predict(self, input_, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0):
        nodes = self.data_handler.nodes
        device = input_[nodes].input_scalar.device
        n_nodes = input_[nodes].input_scalar.shape[0]
        steps = self.num_sampler_steps

        # Karras step schedule
        idx = torch.arange(steps, device=device, dtype=torch.float32)
        sigmas = (sigma_max ** (1 / rho) + idx / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)])

        x = torch.randn(n_nodes, self.data_handler.out_dim, device=device) * sigma_max

        for i in range(steps):
            s_cur, s_next = sigmas[i : i + 1], sigmas[i + 1 : i + 2]

            D_cur = self._denoise(input_, x, s_cur)
            d_cur = (x - D_cur) / s_cur
            x_next = x + (s_next - s_cur) * d_cur

            if s_next.item() > 0:  # Heun 2nd-order correction
                D_next = self._denoise(input_, x_next, s_next)
                d_next = (x_next - D_next) / s_next
                x = x + (s_next - s_cur) * 0.5 * (d_cur + d_next)
            else:
                x = x_next

        return x

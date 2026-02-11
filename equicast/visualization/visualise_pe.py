"""Visualise the sinusoidal positional encoder features."""

import math

import matplotlib.pyplot as plt
import torch

hidden_dim = 32
max_length = 10.0

half = hidden_dim // 2
freq = torch.exp(-torch.arange(half, dtype=torch.float) * math.log(max_length) / half)

dist = torch.linspace(0, max_length, 500)
x = dist.unsqueeze(-1) * freq  # [500, half]
pe = torch.cat([x.sin(), x.cos()], dim=-1)  # [500, hidden_dim]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap of all features
im = axes[0].imshow(
    pe.T.numpy(),
    aspect="auto",
    origin="lower",
    extent=[0, max_length, 0, hidden_dim],
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
)
axes[0].set_xlabel("Distance")
axes[0].set_ylabel("Feature index")
axes[0].set_title("Positional encoding features")
fig.colorbar(im, ax=axes[0])

# A few individual channels
for i in [0, half // 4, half // 2, half - 1]:
    axes[1].plot(dist.numpy(), pe[:, i].numpy(), label=f"sin ch {i}")
for i in [half, half + half // 4, half + half // 2, hidden_dim - 1]:
    axes[1].plot(dist.numpy(), pe[:, i].numpy(), ls="--", label=f"cos ch {i - half}")
axes[1].set_xlabel("Distance")
axes[1].set_ylabel("Value")
axes[1].set_title("Selected channels")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig("positional_encoding.png", dpi=150)
plt.show()
print("Saved to positional_encoding.png")

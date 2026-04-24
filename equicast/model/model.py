# Shim for checkpoint compatibility — old checkpoints reference equicast.model.model.equivariant_loss_fn
from equicast.model.deterministic import Deterministic, equivariant_loss_fn  # noqa: F401

# Backward-compatible aliases
DeterministicModel = Deterministic
Model = Deterministic

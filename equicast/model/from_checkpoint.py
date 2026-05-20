import torch


def load_from_checkpoint(cls, ckpt_path, **kwargs):
    """Load a model from checkpoint, handling torch.compile key remapping."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    ckpt["state_dict"] = {
        k.replace("._orig_mod.", "."): v
        for k, v in ckpt["state_dict"].items()
    }

    from equicast.model.base import BaseModel
    if cls is BaseModel:
        from equicast.model.deterministic import Deterministic
        from equicast.model.diffusion import DiffusionModel
        keys = ckpt["state_dict"].keys()
        cls = DiffusionModel if any("logvar" in k for k in keys) else Deterministic

    hparams = {**ckpt["hyper_parameters"], **kwargs, "compile_backbone": False}
    model = cls(**hparams)
    model.load_state_dict(ckpt["state_dict"])
    return model

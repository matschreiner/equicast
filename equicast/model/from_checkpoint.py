import io

import torch


def load_from_checkpoint(cls, ckpt_path, **kwargs):
    """Load a model from checkpoint, handling torch.compile key remapping.

    If cls is BaseModel, auto-detects Deterministic vs DiffusionModel from state dict keys.
    """
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

    buf = io.BytesIO()
    torch.save(ckpt, buf)
    buf.seek(0)
    return cls.load_from_checkpoint(buf, compile_backbone=False, **kwargs)

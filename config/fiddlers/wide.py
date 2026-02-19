import fiddle as fdl


def fiddler(cfg: fdl.Config) -> None:
    cfg.model.backbone.hidden_dim = 256

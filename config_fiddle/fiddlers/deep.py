import fiddle as fdl


def fiddler(cfg: fdl.Config) -> None:
    cfg.model.backbone.edges = [
        ("grid", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "mesh"),
        ("mesh", "to", "grid"),
    ]

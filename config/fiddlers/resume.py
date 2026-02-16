import fiddle as fdl

from equicast.utils.mlflow_loader import load_checkpoint_path_from_mlflow


def fiddler(cfg: fdl.Config, run_id: str) -> None:
    cfg.ckpt_path = load_checkpoint_path_from_mlflow(run_id)
    cfg.logger.run_id = run_id
    cfg.trainer.logger.run_id = run_id

import fiddle as fdl

from equicast.checkpoint.checkpoint_provider import MLFlowCheckpointProvider


def fiddler(cfg: fdl.Config, run_id: str) -> None:
    provider = MLFlowCheckpointProvider(cfg.logger.tracking_uri, run_id, "latest")
    cfg.ckpt_path = provider.get_checkpoint()
    cfg.logger.run_id = run_id
    cfg.trainer.logger.run_id = run_id

import fiddle as fdl
from fiddle import selectors

from equicast.callbacks import StepTimer
from equicast.data import EquivariantGraphDataHandler, MultiFrameEquivariantGraphDataHandler


def fiddler(cfg: fdl.Config) -> None:
    n_input_frames = 2
    cfg.dataloader.dataset.no_frames = n_input_frames + 1

    new_data_handler = fdl.Config(
        MultiFrameEquivariantGraphDataHandler,
        dataset_path=cfg.model.data_handler.dataset_path,
        feature_config=cfg.model.data_handler.feature_config,
        n_input_frames=n_input_frames,
    )
    selectors.select(cfg, EquivariantGraphDataHandler).replace(new_data_handler)

    cfg.logger.experiment_name = "gnn_vs_painn"

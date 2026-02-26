import fiddle as fdl
from fiddle import selectors

from equicast.callbacks import StepTimer
from equicast.data import EquivariantGraphDataHandler, MultiFrameEquivariantGraphDataHandler

N_INPUT_FRAMES = 2


def fiddler(cfg: fdl.Config) -> None:
    # Dataset: 3 frames total (2 input + 1 target)
    cfg.dataloader.dataset.no_frames = N_INPUT_FRAMES + 1

    # Replace data_handler with MultiFrame variant everywhere in the DAG
    cfg.data_handler = fdl.Config(
        MultiFrameEquivariantGraphDataHandler,
        dataset_path=cfg.model.data_handler.dataset_path,
        feature_config=cfg.model.data_handler.feature_config,
        n_input_frames=N_INPUT_FRAMES,
    )
    selectors.select(cfg, EquivariantGraphDataHandler).replace(new_data_handler)

    cfg.dataloader.num_workers = 0
    cfg.dataloader.shuffle = False
    cfg.trainer.max_epochs = 1
    cfg.trainer.callbacks = [fdl.Config(StepTimer)]
    cfg.logger.experiment_name = "multiframe"

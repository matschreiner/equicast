import fiddle as fdl

from equicast.callbacks import StepTimer
from equicast.data import MultiFrameEquivariantGraphDataHandler

DATASET_PATH = "resources/benchmark.zarr"
GRAPH_PATH = "resources/graph.pt"
N_INPUT_FRAMES = 2


def fiddler(cfg: fdl.Config) -> None:
    # Dataset: 3 frames total (2 input + 1 target)
    cfg.dataloader.dataset.no_frames = N_INPUT_FRAMES + 1

    # Replace data_handler with MultiFrame variant (preserve feature_config reference)
    feature_config = cfg.model.data_handler.feature_config
    new_data_handler = fdl.Config(
        MultiFrameEquivariantGraphDataHandler,
        dataset_path=DATASET_PATH,
        feature_config=feature_config,
        n_input_frames=N_INPUT_FRAMES,
    )
    cfg.model.data_handler = new_data_handler
    cfg.model.backbone.data_handler = new_data_handler
    cfg.model.metrics_tracker.data_handler = new_data_handler

    cfg.dataloader.dataset.path = DATASET_PATH
    cfg.dataloader.dataset.graph_provider.graph_path = GRAPH_PATH
    cfg.dataloader.num_workers = 0
    cfg.dataloader.shuffle = False
    cfg.trainer.max_epochs = 1
    cfg.trainer.callbacks = [fdl.Config(StepTimer)]
    cfg.logger.experiment_name = "multiframe"

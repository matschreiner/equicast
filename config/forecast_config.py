"""Forecast from a trained model checkpoint."""

import base_config
import fiddle as fdl

from config.base_config import (
    get_data_handler,
    get_feature_config,
    get_forecaster,
)
from equicast import experiments
from equicast.model import Model
from equicast.model.backbones.gnn import GNN


def main():
    feature_config = get_feature_config()
    data_handler = get_data_handler(feature_config=feature_config)

    backbone = fdl.Config(
        GNN,
        feature_config,
    )
    graph_provider = base_config.get_graph_provider()

    model = fdl.Config(
        Model,
        backbone=backbone,
        data_handler=data_handler,
        optimizer_factory=None,
        scheduler_factory=None,
    )

    #  checkpoint_provider = base_config.get_checkpoint_provider(
    #      run_id="8bff17dcd64247ce8802a83f0177ad3d",
    #  )
    #  model = base_config.get_model_from_checkpoint(checkpoint_provider)

    forecaster = get_forecaster(model)

    # Setup forecast experiment
    cfg = fdl.Config(
        experiments.ForecastConfig,
        forecaster=forecaster,
        dataset_path="/home/masc/storage/mini_aifs.zarr",
        graph_provider=base_config.get_graph_provider(),
        start_idx=0,
        steps=10,
    )

    experiments.run_experiment(cfg)


if __name__ == "__main__":
    main()

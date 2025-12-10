from dataclasses import dataclass

import fiddle as fdl
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from equicast.checkpoint.checkpoint_provider import MLFlowCheckpointProvider
from equicast.data import FeatureConfig
from equicast.dataset import AnemoiDataset
from equicast.forecaster import Forecaster
from equicast.graph.graph_provider import StaticGraphProvider
from equicast.model import Model
from equicast.model.from_checkpoint import from_checkpoint
from equicast.model.model import Model
from equicast.utils import vis_config


@dataclass
class Experiment:
    forecaster: Forecaster
    dataset: Dataset
    forecast_steps: int

    def run(self) -> torch.Tensor:
        __import__("pdb").set_trace()  # TODO delme

        return self.forecaster(batch, steps)


def make_experiment_config():
    checkpoint_provider = fdl.Config(
        MLFlowCheckpointProvider,
        tracking_uri="https://mlflow.dmidev.org/",
        run_id="8bff17dcd64247ce8802a83f0177ad3d",
        checkpoint_name="latest",
    )
    model = fdl.Config(
        from_checkpoint,
        model_cls=Model,
        checkpoint_provider=checkpoint_provider,
    )

    graph_provider = fdl.Config(
        StaticGraphProvider,
        path="./graph/aifs-single.pt",
    )

    feature_config = FeatureConfig.from_yaml("hydraconfig/features/base.yaml")

    graph_provider = fdl.Config(
        StaticGraphProvider,
        path="./graph/aifs-single.pt",
    )

    dataset = fdl.Config(
        AnemoiDataset,
        path="/home/masc/storage/mini_aifs.zarr",
        feature_config=feature_config,
        graph_provider=graph_provider,
    )

    forecaster = fdl.Config(
        Forecaster,
        model=model,
    )

    return fdl.Config(
        Experiment,
        forecaster=forecaster,
        dataset=dataset,
        forecast_steps=3,
    )


def main():
    experiment_cfg = make_experiment_config()
    vis_config(experiment_cfg)
    experiment = fdl.build(experiment_cfg)
    experiment.run()


main()

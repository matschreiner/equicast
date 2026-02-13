import fiddle as fdl
import torch

from equicast.data import FeatureConfig, GraphDataHandler
from equicast.model.backbones.graphcast import Graphcast
from equicast.model.model import default_loss_fn

torch.set_float32_matmul_precision("medium")


def backbone_config(dataset_path):
    feature_config = fdl.Config(FeatureConfig.from_yaml, path="hydraconfig/features/base.yaml")
    data_handler = fdl.Config(
        GraphDataHandler, feature_config=feature_config, dataset_path=dataset_path
    )

    backbone = fdl.Config(
        Graphcast,
        data_handler,
        num_input_steps=2,
    )

    return backbone, default_loss_fn

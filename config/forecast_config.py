import fiddle as fdl

from config.base_config import get_dataset, get_feature_config, get_graph_provider
from equicast import utils
from equicast.checkpoint.checkpoint_provider import MLFlowCheckpointProvider
from equicast.forecaster import Forecaster
from equicast.model import Model
from equicast.model.from_checkpoint import from_checkpoint


def main():
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

    # Get shared base configurations
    feature_config = get_feature_config()
    graph_provider = get_graph_provider()
    dataset = get_dataset(
        feature_config=feature_config,
        graph_provider=graph_provider,
    )

    forecaster = fdl.Config(
        Forecaster,
        model=model,
    )

    cfg = fdl.Config(
        utils.ForecastConfig,
        forecaster=forecaster,
    )

    utils.vis_config(cfg)


if __name__ == "__main__":
    main()

import fiddle as fdl

from config.base_config import get_data_handler, get_feature_config
from equicast import experiments
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
    data_handler = get_data_handler(feature_config=feature_config)

    forecaster = fdl.Config(
        Forecaster,
        model=model,
        data_handler=data_handler,
    )

    cfg = fdl.Config(
        experiments.ForecastConfig,
        forecaster=forecaster,
    )

    experiments.vis_config(cfg)


if __name__ == "__main__":
    main()

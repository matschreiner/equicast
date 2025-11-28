import sys

from hydra import compose, initialize
from hydra.utils import instantiate


def main(cfg):
    model = instantiate(cfg.model)
    dataset = instantiate(cfg.dataset)

    batch = dataset[0]

    forecaster = instantiate(
        cfg.forecaster,
        model=model,
    )

    forecast = forecaster(batch, steps=5)


if __name__ == "__main__":
    overrides = sys.argv[1:]

    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name="forecast_config", overrides=overrides)

    main(cfg)

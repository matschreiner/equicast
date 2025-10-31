# src/equicast/config/logger_config.py
from dataclasses import dataclass, field
from typing import Optional, Union


@dataclass
class MLFlowLoggerCfg:
    experiment_name: str
    tracking_uri: str
    _target_: str = "pytorch_lightning.loggers.MLFlowLogger"
    run_name: str | None = None


@dataclass
class TensorBoardLoggerCfg:
    save_dir: str
    name: str
    _target_: str = "pytorch_lightning.loggers.TensorBoardLogger"


LoggerConfig = Union[MLFlowLoggerCfg, TensorBoardLoggerCfg, None]

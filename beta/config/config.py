from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class Parameters:
    # Data parameters
    subset: int | None
    lookback: int
    batch_size: int

    # Model parameters
    hidden_size: int
    num_layers: int
    dropout: float


@dataclass
class Experiment:
    dev: bool
    name: str
    version: str | None
    sub_dir: str | None
    min_epochs: int
    max_epochs: int
    progress: bool
    parameters: Parameters


@dataclass
class Config:
    data: Any
    predictor: Any
    experiment: Experiment


OmegaConf.register_new_resolver("len", len)
OmegaConf.register_new_resolver("name", lambda path: Path(path).stem)
ConfigStore.instance().store(name="beta", node=Config)

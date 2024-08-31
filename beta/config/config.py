from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class Run:
    dev: bool
    name: str
    min_epochs: int
    max_epochs: int
    progress: bool


@dataclass
class Experiment:
    # Data parameters
    subset: int | None
    lookback: int
    batch_size: int

    # Model parameters
    hidden_size: int
    num_layers: int
    dropout: float


@dataclass
class Config:
    data: Any
    predictor: Any
    run: Run
    experiment: Experiment


OmegaConf.register_new_resolver("len", len)
ConfigStore.instance().store(name="beta", node=Config)

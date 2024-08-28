from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class Run:
    dev: bool
    min_epochs: int
    max_epochs: int
    progress: bool


@dataclass
class Config:
    data: Any
    predictor: Any
    run: Run


OmegaConf.register_new_resolver("len", len)
ConfigStore.instance().store(name="beta", node=Config)

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore


@dataclass
class Series:
    window: int
    stock: Path
    index: Path
    rates: Any


@dataclass
class Dataset:
    lookback: int
    subset: int | None
    series: Series


@dataclass
class Data:
    dataset: Dataset
    batch_size: int
    shuffle: bool
    split: float


@dataclass
class Model:
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float


@dataclass
class Predictor:
    model: Model
    lr: float


@dataclass
class Experiment:
    dataset_dir: Path
    stock_dir: str
    stock_name: str
    index_dir: str
    index_name: str
    min_epochs: int
    max_epochs: int
    dev_run: bool


@dataclass
class Config:
    data: Data
    predictor: Predictor
    experiment: Experiment


ConfigStore.instance().store(name="beta", node=Config)

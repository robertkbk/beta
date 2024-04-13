from dataclasses import dataclass
from pathlib import Path

from hydra.core.config_store import ConfigStore


@dataclass
class Series:
    index: Path
    stock: Path
    column: str


@dataclass
class Dataset:
    lookback: int
    series: Series
    subset: int | None


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
class Run:
    dev: bool
    min_epochs: int
    max_epochs: int


@dataclass
class Experiment:
    dataset_dir: Path
    stock_dir: str
    stock_name: str
    index_dir: str
    index_name: str


@dataclass
class Config:
    data: Data
    experiment: Experiment
    predictor: Predictor
    run: Run


ConfigStore.instance().store(name="beta", node=Config)

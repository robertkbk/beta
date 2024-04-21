from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class Series:
    index: Path
    stock: Path
    column: Literal["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"]


@dataclass
class Dataset:
    lookback: int
    series: list[Series]
    subset: int | None


@dataclass
class Data:
    dataset: Dataset
    batch_size: int
    predict: int
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
class Config:
    data: Data
    predictor: Predictor
    run: Run


OmegaConf.register_new_resolver("len", len)
ConfigStore.instance().store(name="beta", node=Config)

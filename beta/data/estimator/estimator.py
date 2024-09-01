import abc

import pandas as pd


class BetaEstimator(abc.ABC):
    @abc.abstractmethod
    def estimate(beta: pd.Series) -> pd.Series: ...

    @property
    @abc.abstractmethod
    def superscript(self) -> str: ...


class Blume(BetaEstimator):
    def __init__(self, gamma: float, phi: float) -> None:
        super().__init__()
        self._gamma = gamma
        self._phi = phi

    def estimate(self, beta: pd.Series) -> pd.Series:
        return self._gamma + self._phi * beta

    @property
    def superscript(self) -> str:
        return "B"

import abc

import pandas as pd


class BetaCalculator(abc.ABC):
    def __str__(self) -> str:
        return f"{type(self).__name__}"

    @abc.abstractmethod
    def calculate(rri: pd.Series, rrs: pd.Series) -> pd.Series: ...


class Rolling(BetaCalculator):
    def __init__(self, window: int) -> None:
        super().__init__()
        self._window = window

    def __str__(self) -> str:
        return f"{super().__str__()} ({self._window})"

    def calculate(self, rri: pd.Series, rrs: pd.Series) -> pd.Series:
        stock_cov = rrs.rolling(self._window).cov(rri)
        index_var = rri.rolling(self._window).var()

        return (stock_cov / index_var).dropna()


class Expanding(BetaCalculator):
    def __init__(self, min_periods: int) -> None:
        super().__init__()
        self._min_periods = min_periods

    def calculate(self, rri: pd.Series, rrs: pd.Series) -> pd.Series:
        stock_cov = rrs.expanding(self._min_periods).cov(rri)
        index_var = rri.expanding(self._min_periods).var()

        return (stock_cov / index_var).dropna()


class EWM(BetaCalculator):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self._alpha = alpha

    def __str__(self) -> str:
        return f"{super().__str__()} ($\\alpha={self._alpha}$)"

    def calculate(self, rri: pd.Series, rrs: pd.Series) -> pd.Series:
        stock_cov = rrs.ewm(self._alpha).cov(rri)
        index_var = rri.ewm(self._alpha).var()

        return (stock_cov / index_var).dropna()

import abc
from datetime import date

import pandas as pd


class BetaCalculator(abc.ABC):
    def __str__(self) -> str:
        return f"{type(self).__name__}"

    @abc.abstractmethod
    def calculate(rri: pd.Series, rrs: pd.Series) -> pd.Series: ...

    @property
    @abc.abstractmethod
    def subscript() -> str: ...


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

    @property
    def subscript(self) -> str:
        return f"w={self._window}"


class Expanding(BetaCalculator):
    def __init__(self, min_periods: int, start_date: str | None = None) -> None:
        super().__init__()
        self._min_periods = min_periods

        self._start = None
        if start_date is not None:
            self._start = date.fromisoformat(start_date)

    def calculate(self, rri: pd.Series, rrs: pd.Series) -> pd.Series:
        if self._start is not None:
            rri = rri[self._start :]
            rrs = rrs[self._start :]

        stock_cov = rrs.expanding(self._min_periods).cov(rri)
        index_var = rri.expanding(self._min_periods).var()

        return (stock_cov / index_var).dropna()

    @property
    def subscript(self) -> str:
        return ""


class EWM(BetaCalculator):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self._alpha = alpha

    def __str__(self) -> str:
        return f"{super().__str__()} ($\\alpha={self._alpha}$)"

    def calculate(self, rri: pd.Series, rrs: pd.Series) -> pd.Series:
        index_ewm = rri.ewm(alpha=self._alpha)

        return (index_ewm.cov(rrs) / index_ewm.var()).dropna()

    @property
    def subscript(self) -> str:
        return f"\\alpha={self._alpha}"

import abc

import numpy as np
import pandas as pd


def _log_return_rate(series: pd.Series) -> float:
    return float(np.log(series.iloc[0]) - np.log(series.iloc[-1]))


def _simple_return_rate(series: pd.Series) -> float:
    return float((series.iloc[0] - series.iloc[-1]) / series.iloc[0])


class RatesCalculator(abc.ABC):
    @abc.abstractmethod
    def calculate(self, series: pd.Series) -> pd.Series: ...


class Daily(RatesCalculator):
    def __init__(self) -> None:
        super().__init__()

    def calculate(self, series: pd.Series) -> pd.Series:
        return series.rolling(2).apply(_log_return_rate).dropna()


class Weekly(RatesCalculator):
    def __init__(self, day: int) -> None:
        super().__init__()

        if not (0 <= day <= 4):
            raise ValueError(f"invalid value of day: {day} (should be in range [0; 4])")

        self._day = day

    def calculate(self, series: pd.Series) -> pd.Series:
        filled_series = series.resample("1D").ffill()

        return (
            filled_series[filled_series.index.weekday == self._day]
            .rolling(2)
            .apply(_log_return_rate)
            .dropna()
        )

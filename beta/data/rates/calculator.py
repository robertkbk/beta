import abc
from enum import Enum

import numpy as np
import pandas as pd


def _log_return_rate(series: pd.Series) -> float:
    return float(np.log(series.iloc[0]) - np.log(series.iloc[-1]))


def _simple_return_rate(series: pd.Series) -> float:
    return float((series.iloc[0] - series.iloc[-1]) / series.iloc[0])


class ReturnRate(str, Enum):
    LOG = "LOG"
    SIMPLE = "SIMPLE"


class RatesCalculator(abc.ABC):
    def __init__(self, rate: ReturnRate) -> None:
        super().__init__()

        match rate:
            case ReturnRate.LOG:
                self._rate_calc = _log_return_rate

            case ReturnRate.SIMPLE:
                self._rate_calc = _simple_return_rate

            case _:
                raise ValueError(f"invalid return rate {rate}")

    def __str__(self) -> str:
        return f"{type(self).__name__}"

    @abc.abstractmethod
    def calculate(self, series: pd.Series) -> pd.Series: ...


class Daily(RatesCalculator):
    def __init__(self, rate: ReturnRate) -> None:
        super().__init__(rate)

    def calculate(self, series: pd.Series) -> pd.Series:
        return series.rolling(2).apply(self._rate_calc).dropna()


class Weekly(RatesCalculator):
    def __init__(self, rate: ReturnRate, day: int) -> None:
        super().__init__(rate)

        if not (0 <= day <= 4):
            raise ValueError(f"invalid value of day: {day} (should be in range [0; 4])")

        self._day = day

    def calculate(self, series: pd.Series) -> pd.Series:
        series_filled = series.resample("1D").ffill()

        return (
            series_filled[series_filled.index.weekday == self._day]
            .rolling(2)
            .apply(self._rate_calc)
            .dropna()
        )


class InterWeekly(RatesCalculator):
    def __init__(self, rate: ReturnRate) -> None:
        super().__init__(rate)

    def calculate(self, series: pd.Series) -> pd.Series:
        series_filled = series.resample("1D").ffill()
        mask_iw = series_filled.index.weekday.isin((0, 4))
        series_iw = series_filled[mask_iw]

        if series_iw.index[0].weekday() == 4:
            series_iw = series_iw[1:]

        return series_iw.rolling(2, step=2).apply(self._rate_calc).dropna()


class Monthly(RatesCalculator):
    def __init__(self, start: bool) -> None:
        super().__init__()
        self._use_start = start

    def calculate(self, series: pd.Series) -> pd.Series:
        series_filled = series.resample("1D").ffill()

        date_mask = (
            series_filled.index.is_month_start
            if self._use_start
            else series_filled.index.is_month_end
        )

        return series_filled[date_mask].rolling(2).apply(self._rate_calc).dropna()

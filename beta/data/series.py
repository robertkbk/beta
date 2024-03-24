from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from beta.data.rates.calculator import RatesCalculator


def _read_series(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, usecols=[2, 4, 7], parse_dates=[0], index_col=0  # Date, Open, Close
    )


def _log_return_rate(series: pd.Series) -> float:
    return np.log(series.iloc[0]) - np.log(series.iloc[-1])


def _calculate_return_rate(
    df: pd.DataFrame,
    column: Literal["<OPEN>", "<CLOSE>"],
    calc: Callable[[pd.Series], float] = _log_return_rate,
) -> pd.Series:
    return df[column].rolling(2).apply(calc).dropna()


def _calculate_beta(rri: pd.Series, rrs: pd.Series, window: int) -> pd.Series:
    stock_cov = rrs.rolling(window).apply(lambda rrs: rrs.cov(rri))
    index_var = rri.rolling(window).var()
    return (stock_cov / index_var).dropna()


class BetaSeries:
    def __init__(
        self,
        stock: Path,
        index: Path,
        window: int,
        rates: RatesCalculator,
    ) -> None:
        index_series, stock_series = _read_series(index), _read_series(stock)
        index_rr = rates.calculate(index_series["<CLOSE>"])
        stock_rr = rates.calculate(stock_series["<CLOSE>"])
        beta = _calculate_beta(index_rr, stock_rr, window)

        self.values = beta.values

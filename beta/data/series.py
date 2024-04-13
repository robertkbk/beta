from pathlib import Path

import pandas as pd

from .rates import RatesCalculator
from .index import BetaCalculator

_COL_DATE = "<DATE>"


def _read_series(path: Path, column: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=[_COL_DATE, column],
        parse_dates=[_COL_DATE],
        index_col=_COL_DATE,
    )


class BetaSeries(pd.Series):
    def __init__(
        self,
        stock: Path,
        index: Path,
        column: str,
        rates: RatesCalculator,
        beta: BetaCalculator,
    ) -> None:
        index_series = _read_series(index, column)
        index_rr = rates.calculate(index_series[column])

        stock_series = _read_series(stock, column)
        stock_rr = rates.calculate(stock_series[column])

        series = beta.calculate(index_rr, stock_rr)
        super().__init__(series)

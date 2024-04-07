from pathlib import Path

import pandas as pd

from .rates import RatesCalculator
from .index import BetaCalculator

_COL_DATE = "<DATE>"
_COL_STOCK = "<CLOSE>"


def _read_series(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        usecols=[_COL_DATE, _COL_STOCK],
        parse_dates=[_COL_DATE],
        index_col=_COL_DATE,
    )


class BetaSeries(pd.Series):
    def __init__(
        self,
        stock: Path,
        index: Path,
        rates: RatesCalculator,
        beta: BetaCalculator,
    ) -> None:
        index_series = _read_series(index)
        index_rr = rates.calculate(index_series[_COL_STOCK])

        stock_series = _read_series(stock)
        stock_rr = rates.calculate(stock_series[_COL_STOCK])

        series = beta.calculate(index_rr, stock_rr)
        super().__init__(series)

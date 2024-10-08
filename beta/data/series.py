from pathlib import Path

import pandas as pd

from .estimator import BetaEstimator
from .index import BetaCalculator
from .rates import RatesCalculator

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
        estimator: BetaEstimator | None = None,
    ) -> None:
        index_series = _read_series(index, column)
        index_rr = rates.calculate(index_series[column])

        stock_series = _read_series(stock, column)
        stock_rr = rates.calculate(stock_series[column])

        data = beta.calculate(index_rr, stock_rr)

        if estimator is not None:
            data = estimator.estimate(data)

        super().__init__(data=data, name=self._generate_label(beta, estimator))

    @staticmethod
    def _generate_label(
        beta: BetaCalculator,
        estimator: BetaEstimator | None,
    ):
        label = f"\\beta_{{{beta.subscript}}}"
        if estimator is not None:
            label += f"^{{{estimator.superscript}}}"

        return f"${label}$"

    def __str__(self) -> str:
        return "beta-series"

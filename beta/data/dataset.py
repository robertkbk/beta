import pandas as pd
import torch
from matplotlib.figure import Figure
from torch.utils.data import Dataset

from .series import BetaSeries


class BetaDataset(Dataset):
    def __init__(
        self,
        series: list[BetaSeries],
        lookback: int,
        subset: int | None,
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()

        beta_series = pd.concat(series, axis=1).dropna()
        beta_tensor = torch.tensor(beta_series.values, dtype=dtype)

        if beta_tensor.ndim <= 1:
            beta_tensor.unsqueeze_(1)

        if subset is not None:
            beta_tensor = beta_tensor[-subset:]

        self._dataset = beta_tensor.unfold(0, lookback, 1).mT
        self.series = beta_series

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._dataset[index],
            self._dataset[index + 1],
        )

    def __len__(self) -> int:
        return len(self._dataset) - 1

    def plot(self, *args, **kwargs) -> Figure:
        ax = self.series.plot.line(*args, **kwargs)
        return ax.get_figure()

    def _build_pred_df(
        self, column: int, data: torch.Tensor, pred_len: int
    ) -> pd.DataFrame:
        y = self.series.iloc[:, column][-pred_len:]
        pred = pd.Series(
            data=data, index=self.series.index[-pred_len:], name="Prediction"
        )

        return pd.concat([y, pred], axis=1)

    def plot_prediction(
        self, pred_batch: list[torch.Tensor], *args, **kwargs
    ) -> list[Figure]:
        pred = torch.concat(pred_batch)
        pred_len = len(pred)
        pred_index = self.series.index[-pred_len:]
        pred_df = pd.DataFrame(pred, index=pred_index, columns=self.series.columns)
        pred_ax = pred_df.plot.line(*args, **kwargs)

        pred_col_ax = [
            self._build_pred_df(column, pred_col, pred_len)
            .plot.line(*args, **kwargs)
            .get_figure()
            for column, pred_col in enumerate(pred.unbind(dim=1))
        ]

        return [pred_ax.get_figure(), *pred_col_ax]

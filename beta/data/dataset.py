import torch
from torch.utils.data import Dataset

from .series import BetaSeries


class BetaDataset(Dataset):
    def __init__(
        self,
        series: BetaSeries,
        lookback: int,
        subset: int | None,
        dtype: torch.dtype = torch.float,
    ) -> None:
        super().__init__()

        beta = series.values
        if subset is not None:
            beta = beta[-subset:]

        beta_series = torch.tensor(beta, dtype=dtype)

        if len(beta_series.shape) <= 1:
            beta_series = beta_series.reshape(-1, 1)

        self._dataset = beta_series.unfold(0, lookback, 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._dataset[index].T,
            self._dataset[index + 1].T,
        )

    def __len__(self) -> int:
        return len(self._dataset) - 1

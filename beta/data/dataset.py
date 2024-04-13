import torch
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

        beta = series[0].values
        if subset is not None:
            beta = beta[-subset:]

        beta_tensor = torch.tensor(beta, dtype=dtype)

        if beta_tensor.ndim <= 1:
            beta_tensor.unsqueeze_(1)

        self._dataset = beta_tensor.unfold(0, lookback, 1).mT
        self.series = series

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._dataset[index],
            self._dataset[index + 1],
        )

    def __len__(self) -> int:
        return len(self._dataset) - 1

from collections.abc import Sequence

import lightning as pl
from torch.utils import data

from .dataset import BetaDataset


class BetaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: BetaDataset,
        batch_size: int,
        shuffle: bool,
        split: float,
        test: int,
    ) -> None:
        super().__init__()
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._val_split = split
        self._test = test

        self.dataset = dataset

    def train_dataloader(self) -> data.DataLoader:
        indices = self._train_indices()
        sampler = (
            data.SubsetRandomSampler(indices=indices)
            if self._shuffle
            else data.Subset(indices=indices)
        )

        return data.DataLoader(
            dataset=self.dataset,
            batch_size=self._batch_size,
            sampler=sampler,
        )

    def val_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            dataset=self.dataset,
            batch_size=self._batch_size,
            sampler=self._val_indices(),
        )

    def test_dataloader(self) -> data.DataLoader:
        return data.DataLoader(
            dataset=self.dataset,
            batch_size=self._batch_size,
            sampler=self._test_indices(),
        )

    def predict_dataloader(self) -> data.DataLoader:
        return self.test_dataloader()

    @property
    def _test_split(self) -> int:
        return len(self.dataset) - self._test

    def _train_indices(self) -> Sequence[int]:
        return [*range(self._test_split)]

    def _val_indices(self) -> Sequence[int]:
        return [*range(int(self._test_split * self._val_split), self._test_split)]

    def _test_indices(self) -> Sequence[int]:
        return [*range(self._test_split, len(self.dataset))]

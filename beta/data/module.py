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
        self._split_idx = int((len(dataset) - test) * split)
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

    def _train_indices(self) -> Sequence[int]:
        return [*range(self._split_idx)]

    def _val_indices(self) -> Sequence[int]:
        return [*range(self._split_idx, len(self.dataset) - self._test)]

    def _test_indices(self) -> Sequence[int]:
        return [*range(len(self.dataset) - self._test, len(self.dataset))]

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
        predict: int,
    ) -> None:
        super().__init__()
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._split_idx = int(len(dataset) * split)
        self._predict = predict

        self.dataset = dataset

    def train_dataloader(self) -> data.DataLoader:
        indices = [*range(self._split_idx)]
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
            sampler=range(self._split_idx, len(self.dataset)),
        )

    def test_dataloader(self) -> data.DataLoader:
        dataset_len = len(self.dataset)

        return data.DataLoader(
            dataset=self.dataset,
            batch_size=self._batch_size,
            sampler=range(dataset_len - self._predict, dataset_len),
        )

    def predict_dataloader(self) -> data.DataLoader:
        return self.test_dataloader()

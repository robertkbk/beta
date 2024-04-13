import lightning as pl
import torchmetrics
from torch import Tensor, nn, optim

from .nn import BetaModel


class BetaPredictor(pl.LightningModule):
    def __init__(self, model: BetaModel, lr: float) -> None:
        super().__init__()
        self._lr = lr

        self.model = model
        self.mse = torchmetrics.MeanSquaredError()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.md = torchmetrics.MinkowskiDistance(p=2)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self._lr)

        return (
            [optimizer],
            [
                optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99),
            ],
        )

    def training_step(self, batch: list[Tensor]) -> Tensor:
        x, y = batch
        pred = self.model.forward(x)

        loss = nn.functional.l1_loss(pred, y)
        self.log("train/loss", loss)

        self.mse.forward(pred, y)
        self.mae.forward(pred, y)
        self.md.forward(pred, y)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log("train/mse", self.mse)
        self.log("train/mae", self.mae)
        self.log("train/md", self.md)

    def validation_step(self, batch: list[Tensor]) -> Tensor:
        x, y = batch
        pred = self.model.forward(x)

        loss = nn.functional.mse_loss(pred, y)
        self.log("val/loss", loss)

        self.mse.forward(pred, y)
        self.mae.forward(pred, y)
        self.md.forward(pred, y)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log("val/mse", self.mse)
        self.log("val/mae", self.mae)
        self.log("val/lce", self.md)

    def predict_step(self, batch: list[Tensor]) -> Tensor:
        pred = self.model.forward(batch[0])
        return pred[:, -1]

import lightning as pl
import torchmetrics
from torch import Tensor, nn, optim

from .nn import BetaModel


class BetaPredictor(pl.LightningModule):
    def __init__(self, model: BetaModel, lr: float) -> None:
        super().__init__()
        self._lr = lr
        self._loss = nn.SmoothL1Loss()

        self.model = model

        self._train_metrics = torchmetrics.MetricCollection(
            {"mse": torchmetrics.MeanSquaredError()}, prefix="train/"
        )
        self._val_metrics = torchmetrics.MetricCollection(
            {
                "mae": torchmetrics.MeanAbsoluteError(),
                "mse": torchmetrics.MeanSquaredError(),
                "md": torchmetrics.MinkowskiDistance(p=2.0),
            },
            prefix="val/",
        )

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

        loss = self._loss.forward(pred, y)
        self.log("train/loss", loss)

        self._train_metrics.update(pred, y)
        self.log_dict(self._train_metrics)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self._train_metrics)

    def validation_step(self, batch: list[Tensor]) -> Tensor:
        x, y = batch
        pred = self.model.forward(x)

        loss = self._loss.forward(pred, y)
        self.log("val/loss", loss)

        self._val_metrics.update(pred, y)
        self.log_dict(self._val_metrics)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._val_metrics)

    def predict_step(self, batch: list[Tensor]) -> Tensor:
        pred = self.model.forward(batch[0])
        return pred[:, -1]

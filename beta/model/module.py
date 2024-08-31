import lightning as pl
import torchmetrics
from torch import Tensor, nn, optim

from .nn import BetaModel


class BetaPredictor(pl.LightningModule):
    def __init__(self, model: BetaModel, lr: float) -> None:
        super().__init__()
        self.lr = lr
        self.loss = nn.SmoothL1Loss()
        self.metrics_train = torchmetrics.MetricCollection(
            {
                "rmse": torchmetrics.MeanSquaredError(),
                "mape": torchmetrics.MeanAbsolutePercentageError(),
            },
            prefix="train/",
        )
        self.metrics_val = torchmetrics.MetricCollection(
            {
                "mae": torchmetrics.MeanAbsoluteError(),
                "rmse": torchmetrics.MeanSquaredError(),
                "mape": torchmetrics.MeanAbsolutePercentageError(),
                "smape": torchmetrics.SymmetricMeanAbsolutePercentageError(),
            },
            prefix="val/",
        )
        self.metrics_test = torchmetrics.MetricCollection(
            {
                "mae": torchmetrics.MeanAbsoluteError(),
                "rmse": torchmetrics.MeanSquaredError(),
            },
            prefix="test/",
        )

        self.model = model

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.99)

        return [optimizer], [lr_scheduler]

    def training_step(self, batch: list[Tensor]) -> Tensor:
        x, y = batch
        pred = self.model.forward(x)

        loss = self.loss.forward(pred, y)
        self.log("train/loss", loss)

        self.metrics_train.update(pred, y)
        self.log_dict(self.metrics_train)

        return loss

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.metrics_train)

    def validation_step(self, batch: list[Tensor]) -> Tensor:
        x, y = batch
        pred = self.model.forward(x)

        loss = self.loss.forward(pred, y)
        self.log("val/loss", loss)

        self.metrics_val.update(pred, y)
        self.log_dict(self.metrics_val)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.metrics_val)

    def test_step(self, batch: list[Tensor]) -> Tensor:
        x, y = batch
        pred = self.model.forward(x)

        self.metrics_test.update(pred[:, -1], y[:, -1])
        self.log_dict(self.metrics_test)

        return pred[:, -1]

    def predict_step(self, batch: list[Tensor]) -> Tensor:
        x, _ = batch
        pred = self.model.forward(x)
        return pred[:, -1]

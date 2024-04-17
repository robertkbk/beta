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
                "mse": torchmetrics.MeanSquaredError(),
                "mape": torchmetrics.MeanAbsolutePercentageError(),
            },
            prefix="train/",
        )
        self.metrics_val = torchmetrics.MetricCollection(
            {
                "mae": torchmetrics.MeanAbsoluteError(),
                "mse": torchmetrics.MeanSquaredError(),
                "mape": torchmetrics.MeanAbsolutePercentageError(),
                "smape": torchmetrics.SymmetricMeanAbsolutePercentageError(),
                "wmape": torchmetrics.WeightedMeanAbsolutePercentageError(),
            },
            prefix="val/",
        )

        self.model = model

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)

        return (
            [optimizer],
            [
                optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99),
            ],
        )

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

    def predict_step(self, batch: list[Tensor]) -> Tensor:
        pred = self.model.forward(batch[0])
        return pred[:, -1]

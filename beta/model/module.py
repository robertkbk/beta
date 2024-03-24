import lightning as pl
import torchmetrics
from torch import Tensor, nn, optim

from .nn import BetaModel


class BetaPredictor(pl.LightningModule):
    def __init__(self, model: BetaModel, lr: float) -> None:
        super().__init__()
        self._lr = lr
        self.model = model

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
        self.log(
            "train/epoch/loss", loss, on_epoch=True, on_step=False, reduce_fx="sum"
        )
        return loss

    def validation_step(self, batch: list[Tensor]) -> Tensor:
        x, y = batch
        pred = self.model.forward(x)

        loss = nn.functional.mse_loss(pred, y)
        rmsle = torchmetrics.functional.mean_squared_log_error(pred, y)

        self.log("val/loss", loss)
        self.log("val/rmsle", rmsle)
        return loss

    def predict_step(self, batch: list[Tensor]) -> Tensor:
        pred = self.model.forward(batch[0])
        breakpoint()
        return pred[:, -1]

import hydra
import lightning as pl
import torch
from lightning.pytorch import callbacks
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot

from beta.data.module import BetaDataModule
from beta.model.module import BetaPredictor

from .config import Config


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: Config):
    datamodule: BetaDataModule = hydra.utils.instantiate(config=config.data)
    predictor: BetaPredictor = hydra.utils.instantiate(config=config.predictor)
    logger = TensorBoardLogger(save_dir="logs", name="beta")

    trainer = pl.Trainer(
        fast_dev_run=config.run.dev,
        min_epochs=config.run.min_epochs,
        max_epochs=config.run.max_epochs,
        logger=logger,
        check_val_every_n_epoch=10,
        callbacks=[
            callbacks.EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=10,
            ),
            callbacks.ModelCheckpoint(
                "logs/model",
                filename="beta",
                monitor="val/loss",
                save_weights_only=True,
            ),
        ],
    )

    trainer.fit(model=predictor, datamodule=datamodule)

    # predictor.eval()
    # loader = datamodule.val_dataloader()
    # pred = trainer.predict(predictor, loader, return_predictions=True)
    # pred = torch.concat(pred).squeeze()

    # print(type(pred))
    # print(pred)

    # y = [a[0][0] for a in datamodule.val_dataloader().dataset][-len(pred) :]
    # pyplot.plot(range(len(pred)), pred)
    # pyplot.plot(range(len(pred)), y)
    # pyplot.show()


if __name__ == "__main__":
    main()

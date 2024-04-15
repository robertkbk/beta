import hydra
import lightning as pl
import matplotlib.pyplot as plt
import scienceplots
from lightning.pytorch import callbacks, loggers

from .config import Config
from .data.module import BetaDataModule
from .model.module import BetaPredictor


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: Config):
    datamodule: BetaDataModule = hydra.utils.instantiate(config=config.data)
    predictor: BetaPredictor = hydra.utils.instantiate(config=config.predictor)
    logger = loggers.TensorBoardLogger(save_dir="logs", name="beta")

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

    logger.experiment.add_figure("dataset", datamodule.dataset.plot(xlabel=""))
    trainer.fit(model=predictor, datamodule=datamodule)


if __name__ == "__main__":
    plt.style.use(["science", "no-latex"])
    main()

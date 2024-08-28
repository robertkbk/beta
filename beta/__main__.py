import hydra
import lightning as pl
import matplotlib.pyplot as plt
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
        logger=logger,
        check_val_every_n_epoch=5,
        fast_dev_run=config.run.dev,
        min_epochs=config.run.min_epochs,
        max_epochs=config.run.max_epochs,
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
            callbacks.LearningRateMonitor("epoch"),
            callbacks.LearningRateFinder(min_lr=1e-6, max_lr=1e-1),
        ],
        enable_progress_bar=config.run.progress,
    )

    dataset_fig = datamodule.dataset.plot(xlabel="", figsize=(8, 6))
    logger.experiment.add_figure("dataset", dataset_fig)

    trainer.fit(model=predictor, datamodule=datamodule)

    if not config.run.dev:
        pred = trainer.predict(
            model=predictor,
            datamodule=datamodule,
            return_predictions=True,
            ckpt_path="best",
        )

        pred_figs = datamodule.dataset.plot_prediction(pred, xlabel="", figsize=(8, 6))

        for i, fig in enumerate(pred_figs):
            logger.experiment.add_figure(f"prediction/{i}", fig)


if __name__ == "__main__":
    import scienceplots

    plt.style.use(["science", "no-latex", "grid"])
    main()

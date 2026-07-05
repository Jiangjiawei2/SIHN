import argparse
import os
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

from sihn.config import load_config
from sihn.data.datamodule import SIHNDataModule
from sihn.lightning_module import SIHNLightningModule
from sihn.utils.seed import seed_everything


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(int(cfg["experiment"]["seed"]))
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = SIHNDataModule(cfg)
    module = SIHNLightningModule(cfg)
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="sihn-{epoch:03d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )
    logger = CSVLogger(save_dir=output_dir, name="logs")
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=int(cfg["train"]["epochs"]),
        precision=cfg["train"].get("precision", "32-true"),
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=int(cfg["train"].get("log_every_n_steps", 20)),
    )
    trainer.fit(module, datamodule=datamodule)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

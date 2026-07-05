from __future__ import annotations

import argparse
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    import pytorch_lightning as pl

from sihn.config import load_config
from sihn.data.datamodule import SIHNDataModule
from sihn.lightning_module import SIHNLightningModule


def evaluate(config_path: str, checkpoint_path: str) -> None:
    cfg = load_config(config_path)
    datamodule = SIHNDataModule(cfg)
    module = SIHNLightningModule.load_from_checkpoint(checkpoint_path, cfg=cfg)
    trainer = pl.Trainer(accelerator="auto", devices="auto", logger=False)
    trainer.test(module, datamodule=datamodule)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint)


if __name__ == "__main__":
    main()

"""PyTorch Lightning data module for SIHN experiments."""

from __future__ import annotations

import os

from torch.utils.data import DataLoader

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    import pytorch_lightning as pl

from sihn.data.datasets import build_dataset


class SIHNDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: str | None = None) -> None:
        if stage in {None, "fit"}:
            self.train_set = build_dataset(self.cfg, "train")
            self.val_set = build_dataset(self.cfg, "val")
        if stage in {None, "test", "predict"}:
            self.test_set = build_dataset(self.cfg, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=int(self.cfg["train"]["batch_size"]),
            shuffle=True,
            num_workers=int(self.cfg["train"]["num_workers"]),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=int(self.cfg["train"]["num_workers"]),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=int(self.cfg["train"]["num_workers"]),
            pin_memory=True,
        )

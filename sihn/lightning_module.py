"""Lightning module for SIHN training and evaluation."""

from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import torch

try:
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover - compatibility fallback
    import pytorch_lightning as pl

from sihn.models.losses import build_loss
from sihn.models.sihn import SIHN
from sihn.utils.complex import channels_to_complex, complex_to_channels
from sihn.utils.metrics import lpips_distance, nrmse, psnr, ssim


class SIHNLightningModule(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg
        self.model = SIHN(**cfg["model"])
        self.loss_fn = build_loss(cfg["train"]["loss"])
        self.metric_names = tuple(str(name).lower() for name in cfg.get("eval", {}).get("metrics", ["psnr", "ssim", "nrmse"]))

    def forward(self, kspace: torch.Tensor, mask: torch.Tensor, sensitivity: torch.Tensor | None = None) -> torch.Tensor:
        return self.model(kspace, mask, sensitivity)

    def configure_optimizers(self):
        train_cfg = self.cfg["train"]
        optimizer_name = str(train_cfg.get("optimizer", "adam")).lower()
        if optimizer_name != "adam":
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        return torch.optim.Adam(
            self.parameters(),
            lr=float(train_cfg["lr"]),
            betas=tuple(train_cfg["betas"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )

    def _target_to_channels(self, batch: dict) -> torch.Tensor:
        if "target" not in batch:
            zeros = batch["kspace"].new_zeros(batch["kspace"].shape[0], *batch["kspace"].shape[-2:])
            return complex_to_channels(zeros)

        target = batch["target"].float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if target.shape[1] == 2:
            return target
        target = target.squeeze(1)
        return torch.stack((target, torch.zeros_like(target)), dim=1)

    def _shared_step(self, batch: dict, stage: str) -> torch.Tensor:
        sensitivity = batch.get("sensitivity")
        pred = self(batch["kspace"], batch["mask"], sensitivity)
        target = self._target_to_channels(batch)
        loss = self.loss_fn(pred, target)
        self.log(f"{stage}/loss", loss, prog_bar=True, batch_size=pred.shape[0], sync_dist=True)
        if stage in {"val", "test"}:
            self.log(f"{stage}_loss", loss, prog_bar=False, batch_size=pred.shape[0], sync_dist=True)

        if stage in {"val", "test"}:
            pred_mag = torch.abs(channels_to_complex(pred))
            target_mag = torch.abs(channels_to_complex(target))
            metric_fns = {
                "psnr": psnr,
                "ssim": ssim,
                "nrmse": nrmse,
                "lpips": lpips_distance,
            }
            for name in self.metric_names:
                if name not in metric_fns:
                    raise ValueError(f"Unsupported metric: {name}")
                value = metric_fns[name](pred_mag, target_mag)
                self.log(f"{stage}/{name}", float(value), prog_bar=name in {"psnr", "ssim"}, batch_size=pred.shape[0], sync_dist=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

"""Training losses."""

from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps**2))


def build_loss(name: str) -> nn.Module:
    lowered = name.lower()
    if lowered == "charbonnier":
        return CharbonnierLoss()
    if lowered in {"l1", "mae"}:
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {name}")

"""MRI forward and adjoint operators."""

from __future__ import annotations

import torch
from torch import nn

from sihn.utils.complex import fft2c, ifft2c


def _broadcast_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=target.device, dtype=target.real.dtype)
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(1)
    return mask


def forward_operator(image: torch.Tensor, mask: torch.Tensor, sensitivity: torch.Tensor | None = None) -> torch.Tensor:
    if sensitivity is None:
        kspace = fft2c(image)
    else:
        if image.ndim == sensitivity.ndim - 1:
            image = image.unsqueeze(1)
        kspace = fft2c(sensitivity * image)
    return _broadcast_mask(mask, kspace) * kspace


def adjoint_operator(kspace: torch.Tensor, mask: torch.Tensor, sensitivity: torch.Tensor | None = None) -> torch.Tensor:
    masked = _broadcast_mask(mask, kspace) * kspace
    image = ifft2c(masked)
    if sensitivity is None:
        return image
    return torch.sum(torch.conj(sensitivity) * image, dim=1)


class DataConsistencyStep(nn.Module):
    """One proximal-gradient data-consistency step."""

    def __init__(self, eta: float = 0.1, rho: float = 0.5):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(float(eta)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))

    def forward(
        self,
        z: torch.Tensor,
        b: torch.Tensor,
        u: torch.Tensor,
        kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = forward_operator(z, mask, sensitivity) - kspace
        grad = adjoint_operator(residual, mask, sensitivity) + self.rho * (z - b + u)
        return z - self.eta * grad

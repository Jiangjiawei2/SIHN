"""Complex-valued MRI tensor helpers."""

from __future__ import annotations

import torch


def fft2c(image: torch.Tensor) -> torch.Tensor:
    image = torch.fft.ifftshift(image, dim=(-2, -1))
    kspace = torch.fft.fft2(image, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(kspace, dim=(-2, -1))


def ifft2c(kspace: torch.Tensor) -> torch.Tensor:
    kspace = torch.fft.ifftshift(kspace, dim=(-2, -1))
    image = torch.fft.ifft2(kspace, dim=(-2, -1), norm="ortho")
    return torch.fft.fftshift(image, dim=(-2, -1))


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x):
        if x.ndim >= 3 and x.shape[-1] == 2:
            x = torch.view_as_complex(x.contiguous())
        else:
            raise ValueError("Expected a complex tensor or a trailing real/imag dimension.")
    return torch.stack((x.real, x.imag), dim=1)


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError("Expected tensor shape [B, 2, H, W].")
    return torch.complex(x[:, 0], x[:, 1])


def root_sum_of_squares(x: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    if torch.is_complex(x):
        return torch.sqrt(torch.sum(torch.abs(x) ** 2, dim=dim) + eps)
    return torch.sqrt(torch.sum(x**2, dim=dim) + eps)

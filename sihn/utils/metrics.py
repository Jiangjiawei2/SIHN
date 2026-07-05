"""Evaluation metrics for MRI reconstruction."""

from __future__ import annotations

import math

import numpy as np


_LPIPS_MODEL = None


def to_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if torch.is_complex(x):
                x = torch.abs(x)
            return x.numpy()
    except ModuleNotFoundError:
        pass
    return np.asarray(x)


def psnr(pred, target, data_range: float | None = None) -> float:
    pred_np = to_numpy(pred).astype(np.float64)
    target_np = to_numpy(target).astype(np.float64)
    if data_range is None:
        data_range = float(target_np.max() - target_np.min())
        data_range = data_range if data_range > 0 else 1.0
    mse = float(np.mean((pred_np - target_np) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def nrmse(pred, target, eps: float = 1e-12) -> float:
    pred_np = to_numpy(pred).astype(np.float64)
    target_np = to_numpy(target).astype(np.float64)
    return float(np.linalg.norm(pred_np - target_np) / (np.linalg.norm(target_np) + eps))


def ssim(pred, target, data_range: float | None = None) -> float:
    pred_np = np.squeeze(to_numpy(pred)).astype(np.float64)
    target_np = np.squeeze(to_numpy(target)).astype(np.float64)
    if data_range is None:
        data_range = float(target_np.max() - target_np.min())
        data_range = data_range if data_range > 0 else 1.0
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(target_np, pred_np, data_range=data_range))
    except ModuleNotFoundError:
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        mux = pred_np.mean()
        muy = target_np.mean()
        sigx = pred_np.var()
        sigy = target_np.var()
        sigxy = ((pred_np - mux) * (target_np - muy)).mean()
        return float(((2 * mux * muy + c1) * (2 * sigxy + c2)) / ((mux**2 + muy**2 + c1) * (sigx + sigy + c2)))


def lpips_distance(pred, target) -> float:
    """Compute LPIPS on magnitude images.

    Inputs are converted to three-channel tensors in [-1, 1]. The LPIPS model is
    loaded lazily so PSNR/SSIM-only experiments do not pay the initialization cost.
    """
    try:
        import torch
        import lpips
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install lpips to use eval.metrics: [lpips].") from exc

    global _LPIPS_MODEL
    pred_t = pred.detach() if isinstance(pred, torch.Tensor) else torch.as_tensor(pred)
    target_t = target.detach() if isinstance(target, torch.Tensor) else torch.as_tensor(target)
    device = pred_t.device
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").eval()
    _LPIPS_MODEL = _LPIPS_MODEL.to(device)

    def prepare(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4 and x.shape[1] != 1:
            x = x[:, :1]
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)
        return x.repeat(1, 3, 1, 1) * 2.0 - 1.0

    with torch.no_grad():
        score = _LPIPS_MODEL(prepare(pred_t), prepare(target_t))
    return float(score.mean().detach().cpu())

"""Sampling masks used for accelerated MRI experiments."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _center_slice(length: int, center_fraction: float) -> slice:
    center_lines = max(1, int(round(length * center_fraction)))
    start = (length - center_lines) // 2
    return slice(start, start + center_lines)


def estimate_acceleration(mask: np.ndarray) -> float:
    sampled = float(np.count_nonzero(mask))
    if sampled == 0:
        raise ValueError("Mask has no sampled entries.")
    return float(mask.size / sampled)


def cartesian_random_mask(
    shape: Iterable[int],
    acceleration: int | float,
    center_fraction: float = 0.08,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a column-wise random Cartesian mask."""

    height, width = [int(v) for v in shape]
    rng = np.random.default_rng(seed)
    line_mask = np.zeros(width, dtype=np.float32)

    center = _center_slice(width, center_fraction)
    line_mask[center] = 1.0

    target_lines = max(1, int(round(width / float(acceleration))))
    remaining = max(0, target_lines - int(line_mask.sum()))
    candidates = np.flatnonzero(line_mask == 0)
    if remaining > 0 and candidates.size > 0:
        chosen = rng.choice(candidates, size=min(remaining, candidates.size), replace=False)
        line_mask[chosen] = 1.0
    return np.tile(line_mask[None, :], (height, 1)).astype(np.float32)


def equispaced_mask(
    shape: Iterable[int],
    acceleration: int | float,
    center_fraction: float = 0.08,
) -> np.ndarray:
    """Generate a deterministic Cartesian mask with equispaced outer samples."""

    height, width = [int(v) for v in shape]
    line_mask = np.zeros(width, dtype=np.float32)
    center = _center_slice(width, center_fraction)
    line_mask[center] = 1.0

    target_lines = max(1, int(round(width / float(acceleration))))
    remaining = max(0, target_lines - int(line_mask.sum()))
    candidates = np.flatnonzero(line_mask == 0)
    if remaining > 0 and candidates.size > 0:
        positions = np.linspace(0, candidates.size - 1, remaining)
        line_mask[candidates[np.round(positions).astype(int)]] = 1.0
    return np.tile(line_mask[None, :], (height, 1)).astype(np.float32)


def radial_mask(
    shape: Iterable[int],
    acceleration: int | float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate an approximate 2D radial mask."""

    height, width = [int(v) for v in shape]
    rng = np.random.default_rng(seed)
    mask = np.zeros((height, width), dtype=np.float32)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = int(math.ceil(math.hypot(height, width)))
    num_spokes = max(1, int(round(min(height, width) / float(acceleration))))
    base_angles = np.linspace(0.0, math.pi, num_spokes, endpoint=False)
    jitter = rng.uniform(-math.pi / (num_spokes * 8), math.pi / (num_spokes * 8), size=num_spokes)

    samples = np.linspace(-radius / 2.0, radius / 2.0, radius * 2 + 1)
    for angle in base_angles + jitter:
        ys = np.rint(cy + samples * math.sin(angle)).astype(int)
        xs = np.rint(cx + samples * math.cos(angle)).astype(int)
        valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        mask[ys[valid], xs[valid]] = 1.0

    mask[int(round(cy)), int(round(cx))] = 1.0
    return mask

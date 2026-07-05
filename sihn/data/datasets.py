"""HDF5 MRI slice datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from sihn.utils.complex import fft2c
from sihn.utils.masks import cartesian_random_mask, equispaced_mask, radial_mask


def _require_h5py():
    try:
        import h5py  # type: ignore

        return h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install h5py to read MRI HDF5 datasets.") from exc


def _to_complex_array(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        return array.astype(np.complex64)
    if array.shape[-1] == 2:
        return (array[..., 0] + 1j * array[..., 1]).astype(np.complex64)
    return array.astype(np.float32).astype(np.complex64)


def _make_mask(shape: tuple[int, int], cfg: dict[str, Any], index: int) -> np.ndarray:
    mask_type = str(cfg.get("type", "random")).lower()
    acceleration = cfg.get("acceleration", 4)
    center_fraction = cfg.get("center_fraction", 0.08)
    seed = int(cfg.get("seed", 2024)) + int(index)
    if mask_type == "random":
        return cartesian_random_mask(shape, acceleration, center_fraction, seed)
    if mask_type == "equispaced":
        return equispaced_mask(shape, acceleration, center_fraction)
    if mask_type == "radial":
        return radial_mask(shape, acceleration, seed)
    raise ValueError(f"Unsupported mask type: {mask_type}")


class H5SliceDataset(Dataset):
    """Generic slice dataset for fastMRI, IXI, and MoDL-style HDF5 files."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        challenge: str,
        mask_cfg: dict[str, Any],
        target_key: str | None = None,
    ):
        self.root = Path(root) / split
        self.challenge = challenge
        self.mask_cfg = mask_cfg
        self.target_key = target_key
        self.examples: list[tuple[Path, int]] = []
        self._index_files()

    def _index_files(self) -> None:
        h5py = _require_h5py()
        files = sorted(self.root.glob("*.h5"))
        if not files:
            raise FileNotFoundError(f"No HDF5 files found under {self.root}")
        for path in files:
            with h5py.File(path, "r") as hf:
                key = "kspace" if "kspace" in hf else next(iter(hf.keys()))
                num_slices = int(hf[key].shape[0]) if hf[key].ndim >= 3 else 1
            self.examples.extend((path, idx) for idx in range(num_slices))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        h5py = _require_h5py()
        path, slice_idx = self.examples[index]
        with h5py.File(path, "r") as hf:
            raw_kspace = hf["kspace"][slice_idx] if "kspace" in hf else None
            target = None
            if self.target_key and self.target_key in hf:
                target = hf[self.target_key][slice_idx]
            sensitivity = None
            for key in ("sensitivity", "sensitivity_maps", "sens_maps", "maps"):
                if key in hf:
                    sensitivity = hf[key][slice_idx]
                    break

        if raw_kspace is None:
            if target is None:
                raise KeyError(f"{path} has neither kspace nor target data.")
            image = torch.from_numpy(_to_complex_array(np.asarray(target)))
            kspace = fft2c(image)
        else:
            kspace = torch.from_numpy(_to_complex_array(np.asarray(raw_kspace)))

        spatial_shape = tuple(int(v) for v in kspace.shape[-2:])
        mask = torch.from_numpy(_make_mask(spatial_shape, self.mask_cfg, index))
        if self.challenge == "multicoil" and mask.ndim == 2:
            mask = mask.unsqueeze(0)

        sample: dict[str, torch.Tensor | str | int] = {
            "kspace": kspace,
            "mask": mask,
            "filename": str(path),
            "slice": slice_idx,
        }
        if target is not None:
            target_arr = torch.from_numpy(np.asarray(target).astype(np.float32))
            sample["target"] = target_arr
        if sensitivity is not None:
            sample["sensitivity"] = torch.from_numpy(_to_complex_array(np.asarray(sensitivity)))
        return sample


def build_dataset(cfg: dict[str, Any], split: str) -> H5SliceDataset:
    data_cfg = cfg["data"]
    split_name = data_cfg.get(f"{split}_split", split)
    return H5SliceDataset(
        root=data_cfg["root"],
        split=split_name,
        challenge=data_cfg.get("challenge", "singlecoil"),
        mask_cfg=cfg["mask"],
        target_key=data_cfg.get("target_key"),
    )

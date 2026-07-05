from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sihn.config import load_config
from sihn.models.sihn import SIHN
from sihn.utils.complex import channels_to_complex
from sihn.utils.masks import cartesian_random_mask, equispaced_mask, radial_mask


def _mask(shape: tuple[int, int], cfg: dict) -> torch.Tensor:
    mask_type = cfg["mask"]["type"]
    acceleration = cfg["mask"]["acceleration"]
    center_fraction = cfg["mask"].get("center_fraction", 0.08)
    seed = cfg["mask"].get("seed", 2024)
    if mask_type == "random":
        arr = cartesian_random_mask(shape, acceleration, center_fraction, seed)
    elif mask_type == "equispaced":
        arr = equispaced_mask(shape, acceleration, center_fraction)
    elif mask_type == "radial":
        arr = radial_mask(shape, acceleration, seed)
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")
    return torch.from_numpy(arr)


def infer(config_path: str, checkpoint_path: str, input_npy: str, output_npy: str) -> None:
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kspace_np = np.load(input_npy)
    kspace = torch.from_numpy(kspace_np.astype(np.complex64)).unsqueeze(0).to(device)
    mask = _mask(tuple(kspace.shape[-2:]), cfg).unsqueeze(0).to(device)

    model = SIHN(**cfg["model"]).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
    if any(key.startswith("model.") for key in state_dict):
        state_dict = {key.removeprefix("model."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        pred = model(kspace, mask)
        image = torch.abs(channels_to_complex(pred)).squeeze(0).cpu().numpy()
    Path(output_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_npy, image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-npy", required=True)
    parser.add_argument("--output-npy", required=True)
    args = parser.parse_args()
    infer(args.config, args.checkpoint, args.input_npy, args.output_npy)


if __name__ == "__main__":
    main()
